import bisect
import math
from datetime import date, timedelta
from functools import lru_cache

DAYS_PER_YEAR = 365.2425
MAX_AGE = 125  # survival is cut off here; it is effectively zero well before
STEP = 1 / 12  # the survival curve is integrated in monthly steps
GOMPERTZ_SCALE = 9.5  # years for the hazard to grow by a factor of e
PERCENTILES = (10, 25, 50, 75, 90)  # of age at death, given survival to the current age

# Overlapping risks (smoking, diabetes, obesity) do not multiply cleanly, so the summed log
# hazard ratio is discounted, then the total is clamped to a plausible range.
OVERLAP_DISCOUNT = 0.85
HAZARD_RATIO_RANGE = (0.3, 8.0)

# Relative risks fade with age: full strength up to FADE_START, half strength from FADE_END.
FADE_START, FADE_END, FADE_FLOOR = 60, 90, 0.5

# Life expectancy at birth (male, female), rounded recent estimates. They only anchor the
# baseline curve for a country; they are not a substitute for its national life table.
LIFE_EXPECTANCY = {
    "world": (70.9, 76.0),
    "united states": (76.3, 81.4),
    "united kingdom": (79.0, 82.8),
    "canada": (80.2, 84.2),
    "australia": (81.3, 85.4),
    "germany": (78.5, 83.4),
    "france": (79.7, 85.6),
    "japan": (81.5, 87.6),
    "china": (75.0, 80.5),
    "india": (69.5, 72.0),
    "brazil": (72.5, 79.3),
    "mexico": (71.0, 77.0),
    "russia": (68.0, 78.0),
    "south africa": (60.0, 66.0),
    "nigeria": (53.5, 55.5),
}

SEXES = ("male", "female")

# Hazard ratios against the average person in the population (1.0).
SMOKING = {"never": 0.85, "former": 1.1, "current": 2.0, "heavy": 2.6}
ALCOHOL = {"none": 1.0, "light": 1.0, "moderate": 1.05, "heavy": 1.5}
EXERCISE = {"sedentary": 1.3, "light": 1.0, "active": 0.8, "very_active": 0.75}
CONDITIONS = {
    "diabetes": 1.8,
    "hypertension": 1.3,
    "heart_disease": 2.0,
    "stroke": 2.0,
    "cancer_history": 1.5,
    "copd": 2.2,
    "kidney_disease": 2.0,
}
BMI_BANDS = (  # (upper bound, hazard ratio)
    (18.5, 1.3),
    (25, 0.95),
    (30, 1.0),
    (35, 1.2),
    (40, 1.5),
    (math.inf, 1.9),
)

DISCLAIMER = (
    "This is a statistical estimate from a simplified model, not a medical prediction. "
    "It cannot say when any individual will die."
)


def list_countries():
    return sorted(LIFE_EXPECTANCY)


def bmi_from(height_cm, weight_kg):
    if height_cm <= 0 or weight_kg <= 0:
        raise ValueError("height_cm and weight_kg must be positive")
    return weight_kg / (height_cm / 100) ** 2


def _choice(name, value, table):
    key = str(value).strip().lower().replace(" ", "_")
    if key not in table:
        raise ValueError(f"{name} must be one of {list(table)}, got {value!r}")
    return key


def _age_of(profile, today):
    if profile.get("birth_date") is not None:
        birth = profile["birth_date"]
        if birth > today:
            raise ValueError("birth_date is in the future")
        age = (today - birth).days / DAYS_PER_YEAR
    elif profile.get("age") is not None:
        age = float(profile["age"])
    else:
        raise ValueError("profile needs an age or a birth_date")
    if not 0 <= age < MAX_AGE:
        raise ValueError(f"age must be between 0 and {MAX_AGE}, got {age:.1f}")
    return age


def risk_factors(profile):
    """The hazard ratio of each supplied risk factor, e.g. {"smoking: current": 2.0}.

    Factors that are left out of the profile contribute nothing (an average person).
    """
    factors = {}
    if profile.get("smoking") is not None:
        key = _choice("smoking", profile["smoking"], SMOKING)
        factors[f"smoking: {key}"] = SMOKING[key]
    if profile.get("alcohol") is not None:
        key = _choice("alcohol", profile["alcohol"], ALCOHOL)
        factors[f"alcohol: {key}"] = ALCOHOL[key]
    if profile.get("exercise") is not None:
        key = _choice("exercise", profile["exercise"], EXERCISE)
        factors[f"exercise: {key}"] = EXERCISE[key]

    bmi = profile.get("bmi")
    if bmi is None and profile.get("height_cm") and profile.get("weight_kg"):
        bmi = bmi_from(profile["height_cm"], profile["weight_kg"])
    if bmi is not None:
        if not 10 <= bmi <= 80:
            raise ValueError(f"bmi {bmi:.1f} is outside the plausible range 10-80")
        ratio = next(hr for upper, hr in BMI_BANDS if bmi < upper)
        factors[f"bmi: {bmi:.1f}"] = ratio

    for condition in profile.get("conditions") or ():
        key = _choice("condition", condition, CONDITIONS)
        factors[f"condition: {key}"] = CONDITIONS[key]
    return factors


def combined_hazard_ratio(factors):
    """Fold per-factor hazard ratios into one, discounting for overlap and clamping."""
    log_total = OVERLAP_DISCOUNT * sum(math.log(hr) for hr in factors.values())
    low, high = HAZARD_RATIO_RANGE
    return min(high, max(low, math.exp(log_total)))


def _fade(age):
    if age <= FADE_START:
        return 1.0
    if age >= FADE_END:
        return FADE_FLOOR
    return 1 - (1 - FADE_FLOOR) * (age - FADE_START) / (FADE_END - FADE_START)


def _gompertz(age, mode):
    return math.exp((age - mode) / GOMPERTZ_SCALE) / GOMPERTZ_SCALE


def _plain_life_expectancy(mode):
    survival, total, age = 1.0, 0.0, 0.0
    while age < MAX_AGE:
        next_survival = survival * math.exp(-_gompertz(age + STEP / 2, mode) * STEP)
        total += (survival + next_survival) / 2 * STEP
        survival, age = next_survival, age + STEP
    return total


@lru_cache(maxsize=None)
def baseline_mode(country, sex):
    """The Gompertz modal age at death that reproduces the country's life expectancy at birth."""
    target = LIFE_EXPECTANCY[country][SEXES.index(sex)]
    low, high = 30.0, 110.0
    for _ in range(40):
        mid = (low + high) / 2
        if _plain_life_expectancy(mid) < target:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def survival_curve(age, mode, hazard_ratio):
    """Ages and survival probabilities on a monthly grid from `age`, conditional on being alive.

    Returns (ages, cumulative_hazard, survival), three lists of equal length.
    """
    steps = int((MAX_AGE - age) / STEP) + 1
    ages = [age + i * STEP for i in range(steps)]
    cumulative = [0.0]
    for i in range(1, steps):
        mid = ages[i] - STEP / 2
        rate = _gompertz(mid, mode) * hazard_ratio ** _fade(mid)
        cumulative.append(cumulative[-1] + rate * STEP)
    return ages, cumulative, [math.exp(-h) for h in cumulative]


def _quantile_age(ages, cumulative, p):
    """The age by which a fraction p of people alive at ages[0] have died."""
    target = -math.log(1 - p)
    i = bisect.bisect_left(cumulative, target)
    if i >= len(ages):
        return float(MAX_AGE)
    if i == 0:
        return ages[0]
    span = cumulative[i] - cumulative[i - 1]
    return ages[i - 1] + STEP * (target - cumulative[i - 1]) / span


def predict_mortality(profile, today=None):
    """Estimate the distribution of age and date at death for a person.

    `profile` is a dict:
      sex           "male" or "female" (required)
      age           years, or birth_date (a `datetime.date`); one is required
      country       a name from `list_countries()` (default "world")
      smoking       never | former | current | heavy
      alcohol       none | light | moderate | heavy
      exercise      sedentary | light | active | very_active
      bmi           or height_cm and weight_kg
      conditions    any of CONDITIONS, e.g. ["diabetes", "hypertension"]
    Anything omitted counts as an average person for that factor.

    The model is a Gompertz hazard anchored to the country's life expectancy, scaled by the
    combined risk-factor hazard ratio, which fades at older ages. The result gives the expected
    and percentile ages at death (given survival to today) and the matching calendar dates.
    """
    today = today or date.today()
    sex = _choice("sex", profile.get("sex", ""), dict.fromkeys(SEXES))
    country = str(profile.get("country", "world")).strip().lower().replace("_", " ")
    if country not in LIFE_EXPECTANCY:
        raise ValueError(f"country must be one of {list_countries()}, got {country!r}")
    age = _age_of(profile, today)

    factors = risk_factors(profile)
    hazard_ratio = combined_hazard_ratio(factors)
    mode = baseline_mode(country, sex)
    ages, cumulative, survival = survival_curve(age, mode, hazard_ratio)

    expected = age + sum(
        (survival[i] + survival[i + 1]) / 2 * STEP for i in range(len(survival) - 1)
    )
    percentiles = {p: _quantile_age(ages, cumulative, p / 100) for p in PERCENTILES}
    per_year = round(1 / STEP)
    return {
        "age": age,
        "sex": sex,
        "country": country,
        "factors": factors,
        "hazard_ratio": hazard_ratio,
        "baseline_life_expectancy": LIFE_EXPECTANCY[country][SEXES.index(sex)],
        "expected_age_at_death": expected,
        "percentile_ages": percentiles,
        "percentile_dates": {
            p: today + timedelta(days=(a - age) * DAYS_PER_YEAR) for p, a in percentiles.items()
        },
        "expected_date": today + timedelta(days=(expected - age) * DAYS_PER_YEAR),
        "survival": [(ages[i], survival[i]) for i in range(0, len(ages), per_year)],
    }


def survival_probability(result, years):
    """Chance of being alive `years` from today, from a `predict_mortality` result."""
    if years < 0:
        raise ValueError("years must not be negative")
    curve = result["survival"]
    start = curve[0][0]
    target = start + years
    if target >= curve[-1][0]:
        return curve[-1][1]
    i = int(years)
    (a0, s0), (a1, s1) = curve[i], curve[i + 1]
    # Survival is exponential in cumulative hazard, so interpolate in log space.
    return math.exp(math.log(s0) + (math.log(s1) - math.log(s0)) * (target - a0) / (a1 - a0))


def format_prediction(result):
    """Render a `predict_mortality` result as text."""
    lines = [
        f"{result['sex'].title()}, age {result['age']:.1f}, {result['country'].title()} "
        f"(baseline life expectancy at birth {result['baseline_life_expectancy']})"
    ]
    if result["factors"]:
        lines.append("Risk factors:")
        lines += [f"  {name}: x{hr:g}" for name, hr in result["factors"].items()]
    lines.append(f"Combined hazard ratio: x{result['hazard_ratio']:.2f}")
    lines += [
        "",
        f"Expected age at death: {result['expected_age_at_death']:.1f} "
        f"({result['expected_date']:%d %b %Y})",
    ]
    for p, years in result["percentile_ages"].items():
        label = "median" if p == 50 else f"{p}th percentile"
        lines.append(f"  {label}: age {years:.1f} ({result['percentile_dates'][p]:%d %b %Y})")
    lines += ["", DISCLAIMER]
    return "\n".join(lines)


def _ask(prompt, parse, optional=False):
    """Prompt until `parse` accepts the answer. A blank answer gives None if optional."""
    while True:
        answer = input(prompt + (" (enter to skip)" if optional else "") + ": ").strip()
        if not answer and optional:
            return None
        try:
            return parse(answer)
        except ValueError as error:
            print(f"  {error}")


def _option(name, table):
    def parse(answer):
        return _choice(name, answer, table)
    return parse


def _number(low, high):
    def parse(answer):
        value = float(answer)
        if not low <= value <= high:
            raise ValueError(f"must be between {low} and {high}")
        return value
    return parse


def _country(answer):
    if answer.lower() not in LIFE_EXPECTANCY:
        raise ValueError("unknown country")
    return answer.lower()


def _conditions(answer):
    names = [name for name in answer.replace(",", " ").split()]
    for name in names:
        _choice("condition", name, CONDITIONS)
    return names


def _birth_date(answer):
    try:
        return date.fromisoformat(answer)
    except ValueError:
        raise ValueError("use the format YYYY-MM-DD") from None


def main():
    profile = {}
    profile["sex"] = _ask(f"Sex {list(SEXES)}", _option("sex", dict.fromkeys(SEXES)))
    born = _ask("Birth date YYYY-MM-DD", _birth_date, optional=True)
    if born:
        profile["birth_date"] = born
    else:
        profile["age"] = _ask("Age in years", _number(0, MAX_AGE - 1))
    profile["country"] = _ask(f"Country {list_countries()}", _country, optional=True) or "world"
    profile["smoking"] = _ask(f"Smoking {list(SMOKING)}", _option("smoking", SMOKING), True)
    profile["alcohol"] = _ask(f"Alcohol {list(ALCOHOL)}", _option("alcohol", ALCOHOL), True)
    profile["exercise"] = _ask(f"Exercise {list(EXERCISE)}", _option("exercise", EXERCISE), True)
    profile["height_cm"] = _ask("Height in cm", _number(50, 260), True)
    if profile["height_cm"]:
        profile["weight_kg"] = _ask("Weight in kg", _number(10, 400))
    profile["conditions"] = _ask(f"Conditions, space separated {list(CONDITIONS)}", _conditions, True)

    print()
    print(format_prediction(predict_mortality(profile)))


if __name__ == "__main__":
    main()
