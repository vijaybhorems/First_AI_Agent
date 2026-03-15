#!/usr/bin/env python3
"""
Travel Planner MCP Server

Exposes 8 travel planning tools:
  - search_flights
  - search_hotels
  - get_weather_forecast
  - find_attractions
  - get_travel_requirements
  - get_currency_info
  - create_itinerary
  - estimate_trip_cost
"""

import asyncio
import json
import random
from datetime import datetime, timedelta
from typing import Any

import mcp.server.stdio
import mcp.types as types
from mcp.server import Server

server = Server("travel-planner")

# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

AIRLINES = {
    "domestic": ["Delta", "United", "American", "Southwest", "JetBlue", "Alaska Airlines"],
    "international": [
        "British Airways", "Lufthansa", "Air France", "Emirates",
        "Singapore Airlines", "Cathay Pacific", "Delta", "United",
        "KLM", "Turkish Airlines",
    ],
}

HOTEL_CHAINS = [
    "Marriott", "Hilton", "Hyatt", "IHG Holiday Inn", "Four Seasons",
    "Ritz-Carlton", "W Hotels", "Westin", "Sheraton", "Accor",
]

ATTRACTION_DATA: dict[str, dict[str, list[str]]] = {
    "Paris": {
        "museums":    ["Louvre Museum", "Musée d'Orsay", "Centre Pompidou", "Musée Rodin"],
        "historical": ["Eiffel Tower", "Notre-Dame Cathedral", "Arc de Triomphe", "Palace of Versailles"],
        "restaurants":["Le Jules Verne", "L'Ami Louis", "Septime", "Frenchie"],
        "shopping":   ["Champs-Élysées", "Le Marais", "Galeries Lafayette", "Rue du Faubourg Saint-Honoré"],
        "nightlife":  ["Moulin Rouge", "Crazy Horse", "Le Baron", "Social Club"],
        "nature":     ["Luxembourg Gardens", "Bois de Boulogne", "Tuileries Garden", "Buttes-Chaumont"],
    },
    "Tokyo": {
        "museums":    ["Tokyo National Museum", "teamLab Borderless", "Mori Art Museum", "Edo-Tokyo Museum"],
        "historical": ["Senso-ji Temple", "Imperial Palace", "Meiji Shrine", "Nikko Toshogu"],
        "restaurants":["Sukiyabashi Jiro", "Narisawa", "Den", "RyuGin"],
        "shopping":   ["Shibuya 109", "Harajuku Takeshita St", "Akihabara", "Ginza Six"],
        "nightlife":  ["Roppongi Hills", "Shinjuku Golden Gai", "Nakameguro", "Shimokitazawa"],
        "nature":     ["Shinjuku Gyoen", "Ueno Park", "Yanaka Cemetery", "Mount Takao"],
    },
    "New York": {
        "museums":    ["Metropolitan Museum of Art", "MoMA", "American Museum of Natural History", "Whitney Museum"],
        "historical": ["Statue of Liberty", "Ellis Island", "Brooklyn Bridge", "Empire State Building"],
        "restaurants":["Le Bernardin", "Eleven Madison Park", "Peter Luger Steak House", "Katz's Deli"],
        "shopping":   ["Fifth Avenue", "SoHo", "Chelsea Market", "Brooklyn Flea"],
        "nightlife":  ["Brooklyn Mirage", "Output", "Le Bain", "Beauty & Essex"],
        "nature":     ["Central Park", "The High Line", "Brooklyn Botanic Garden", "Prospect Park"],
    },
    "Barcelona": {
        "museums":    ["Picasso Museum", "MNAC", "Fundació Joan Miró", "Museu d'Art Contemporani"],
        "historical": ["Sagrada Família", "Park Güell", "Gothic Quarter", "Montjuïc Castle"],
        "restaurants":["El Bulli", "Tickets", "Cinc Sentits", "La Boqueria Market"],
        "shopping":   ["La Rambla", "Passeig de Gràcia", "El Born", "Barceloneta Market"],
        "nightlife":  ["Opium", "CDLC", "Razzmatazz", "Sala Apolo"],
        "nature":     ["Barceloneta Beach", "Parc de la Ciutadella", "Montjuïc Hill", "Tibidabo"],
    },
    "Bali": {
        "museums":    ["Agung Rai Museum of Art", "Neka Art Museum", "Museum Puri Lukisan"],
        "historical": ["Tanah Lot Temple", "Uluwatu Temple", "Besakih Temple", "Tirta Empul"],
        "restaurants":["Locavore", "Sarong", "Mozaic", "Merah Putih"],
        "shopping":   ["Seminyak Square", "Ubud Art Market", "Kuta Beach Walk", "Seminyak Village"],
        "nightlife":  ["Sky Garden", "Potato Head Beach Club", "Ku De Ta", "Mirror"],
        "nature":     ["Tegalalang Rice Terraces", "Mount Batur", "Gitgit Waterfall", "Nusa Penida"],
    },
}

WEATHER_PATTERNS = {
    "tropical":      {"temp_high": (28, 35), "temp_low": (22, 28), "conditions": ["Sunny", "Partly Cloudy", "Thunderstorms", "Humid & Clear"]},
    "mediterranean": {"temp_high": (20, 30), "temp_low": (12, 20), "conditions": ["Sunny", "Clear", "Partly Cloudy", "Mild Breeze"]},
    "continental":   {"temp_high": (15, 25), "temp_low": (5, 15),  "conditions": ["Cloudy", "Partly Cloudy", "Light Rain", "Clear"]},
    "cold":          {"temp_high": (-5, 10), "temp_low": (-15, 0), "conditions": ["Snow", "Overcast", "Cold & Windy", "Freezing"]},
}

CITY_WEATHER_PATTERN = {
    "Paris": "continental", "London": "continental",
    "Tokyo": "continental", "New York": "continental",
    "Bali": "tropical",     "Bangkok": "tropical",
    "Barcelona": "mediterranean", "Rome": "mediterranean",
    "Sydney": "mediterranean",    "Dubai": "tropical",
    "Reykjavik": "cold",    "Oslo": "cold",
}

VISA_DATA: dict[tuple[str, str], dict] = {
    ("US", "France"):    {"required": False, "notes": "No visa for stays ≤90 days (Schengen). Passport valid 6+ months."},
    ("US", "Japan"):     {"required": False, "notes": "No visa for stays ≤90 days. Passport valid for duration of stay."},
    ("US", "UK"):        {"required": False, "notes": "No visa for stays ≤6 months. Electronic Travel Authorisation (ETA) required from Jan 2025."},
    ("US", "Australia"): {"required": True,  "type": "eVisitor (651)", "cost": "Free", "notes": "Apply online before travel. Usually approved instantly."},
    ("US", "India"):     {"required": True,  "type": "e-Visa",         "cost": "$25",  "notes": "Apply online ≥4 business days before arrival."},
    ("US", "Brazil"):    {"required": False, "notes": "Visa-free since June 2024 for stays ≤90 days."},
    ("US", "Thailand"):  {"required": False, "notes": "Visa exemption ≤30 days (extendable once). Passport valid 6+ months."},
    ("US", "Mexico"):    {"required": False, "notes": "No visa needed. Tourist card (FMM) required; often included in airfare."},
    ("UK", "France"):    {"required": False, "notes": "No visa (post-Brexit) for stays ≤90 days within any 180-day period."},
    ("US", "Spain"):     {"required": False, "notes": "No visa for stays ≤90 days (Schengen). Passport valid 6+ months."},
}

CURRENCY_DATA: dict[str, dict] = {
    "France":    {"currency": "Euro",            "code": "EUR", "rate_to_usd": 1.08,  "tips": ["Cards widely accepted", "ATMs everywhere", "Tipping 5–10% optional"]},
    "Japan":     {"currency": "Japanese Yen",    "code": "JPY", "rate_to_usd": 0.0067,"tips": ["Cash preferred", "Use 7-Eleven/Japan Post ATMs", "No tipping culture"]},
    "UK":        {"currency": "British Pound",   "code": "GBP", "rate_to_usd": 1.27,  "tips": ["Cards accepted widely", "Tipping 10–15% in restaurants", "Contactless common"]},
    "Australia": {"currency": "Australian Dollar","code": "AUD", "rate_to_usd": 0.65, "tips": ["Cards accepted everywhere", "No strong tipping culture", "ATMs widely available"]},
    "Thailand":  {"currency": "Thai Baht",       "code": "THB", "rate_to_usd": 0.028, "tips": ["Mix of cash and cards", "ATMs charge ~220 THB fee", "Tipping appreciated"]},
    "Mexico":    {"currency": "Mexican Peso",    "code": "MXN", "rate_to_usd": 0.058, "tips": ["Cash preferred locally", "Tipping 10–15%", "USD accepted in tourist areas"]},
    "India":     {"currency": "Indian Rupee",    "code": "INR", "rate_to_usd": 0.012, "tips": ["Carry cash for markets", "Cards in major cities", "Tipping 10% appreciated"]},
    "UAE":       {"currency": "UAE Dirham",      "code": "AED", "rate_to_usd": 0.27,  "tips": ["Cards accepted everywhere", "Tipping 10–15%", "ATMs widely available"]},
    "Spain":     {"currency": "Euro",            "code": "EUR", "rate_to_usd": 1.08,  "tips": ["Cards widely accepted", "Tipping 5–10% optional", "ATMs in all towns"]},
    "Indonesia": {"currency": "Indonesian Rupiah","code":"IDR", "rate_to_usd": 0.000063, "tips": ["Cash for local vendors", "ATMs in tourist areas", "Tipping appreciated not required"]},
}

DAILY_COST_USD: dict[str, dict[str, int]] = {
    "Paris":     {"budget": 150, "moderate": 300, "luxury": 600},
    "Tokyo":     {"budget": 120, "moderate": 250, "luxury": 500},
    "New York":  {"budget": 200, "moderate": 350, "luxury": 700},
    "Bali":      {"budget":  60, "moderate": 150, "luxury": 400},
    "Bangkok":   {"budget":  50, "moderate": 120, "luxury": 350},
    "Barcelona": {"budget": 100, "moderate": 200, "luxury": 500},
    "London":    {"budget": 180, "moderate": 320, "luxury": 650},
    "Sydney":    {"budget": 150, "moderate": 280, "luxury": 550},
    "Dubai":     {"budget": 120, "moderate": 250, "luxury": 700},
    "Rome":      {"budget":  90, "moderate": 180, "luxury": 450},
}
DEFAULT_DAILY_COST: dict[str, int] = {"budget": 80, "moderate": 180, "luxury": 400}

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _search_flights(args: dict) -> dict:
    origin      = args["origin"]
    destination = args["destination"]
    dep_date    = args["departure_date"]
    ret_date    = args.get("return_date")
    passengers  = int(args.get("passengers", 1))
    cabin       = args.get("cabin_class", "economy")

    is_intl = not (origin.lower() in ("jfk", "lax", "ord", "atl", "dfw", "sfo") and
                   destination.lower() in ("jfk", "lax", "ord", "atl", "dfw", "sfo"))
    pool = AIRLINES["international"] if is_intl else AIRLINES["domestic"]

    base = {"economy": random.randint(280, 850),
            "business": random.randint(1_400, 4_500),
            "first": random.randint(5_000, 15_000)}[cabin]

    def make_flights(date: str, n: int) -> list[dict]:
        flights = []
        for _ in range(n):
            airline  = random.choice(pool)
            price    = int(base * random.uniform(0.75, 1.45))
            dep_h    = random.randint(5, 22)
            dur_h    = random.randint(2, 15)
            stops    = random.choices([0, 1, 2], weights=[55, 35, 10])[0]
            dep_min  = random.choice(["00", "15", "30", "45"])
            arr_h    = (dep_h + dur_h) % 24
            arr_min  = random.choice(["00", "10", "25", "40"])
            flights.append({
                "flight_id": f"{airline[:2].upper()}{random.randint(100, 999)}",
                "airline": airline,
                "departure": f"{date} {dep_h:02d}:{dep_min}",
                "arrival":   f"{date} {arr_h:02d}:{arr_min}",
                "duration":  f"{dur_h}h {random.choice([0, 15, 30, 45])}m",
                "stops":     stops,
                "stopover_cities": [random.choice(["Chicago", "London", "Frankfurt", "Dubai", "Doha"])] if stops else [],
                "price_per_person": price,
                "total_price": price * passengers,
                "cabin": cabin,
                "baggage": "1 checked bag included" if cabin != "economy" else "Carry-on only",
                "refundable": random.choice([True, False]),
                "seats_left": random.randint(2, 18),
            })
        flights.sort(key=lambda x: x["price_per_person"])
        return flights

    result: dict[str, Any] = {
        "route": f"{origin.upper()} → {destination.upper()}",
        "departure_date": dep_date,
        "passengers": passengers,
        "cabin_class": cabin,
        "outbound_flights": make_flights(dep_date, random.randint(4, 6)),
    }
    if ret_date:
        result["return_date"] = ret_date
        result["return_flights"] = make_flights(ret_date, random.randint(3, 5))
    return result


def _search_hotels(args: dict) -> dict:
    city       = args["city"]
    check_in   = args["check_in"]
    check_out  = args["check_out"]
    guests     = int(args.get("guests", 1))
    max_price  = args.get("max_price_per_night")
    min_stars  = int(args.get("star_rating", 0))

    try:
        nights = max(1, (datetime.strptime(check_out, "%Y-%m-%d") -
                         datetime.strptime(check_in,  "%Y-%m-%d")).days)
    except ValueError:
        nights = 3

    amenity_pool = [
        "Free WiFi", "Outdoor Pool", "Fitness Centre", "Spa & Wellness",
        "On-site Restaurant", "Rooftop Bar", "24h Room Service", "Concierge",
        "Valet Parking", "Airport Shuttle", "Business Centre", "Pet Friendly",
    ]
    price_range = {2: (60, 110), 3: (110, 210), 4: (210, 420), 5: (420, 1_300)}

    hotels = []
    for _ in range(random.randint(5, 8)):
        stars = random.randint(max(min_stars, 2), 5)
        lo, hi = price_range[stars]
        ppn = random.randint(lo, hi)
        if max_price and ppn > max_price:
            continue
        suffixes = ["Grand", "Plaza", "Royal", "Palace", "Boutique", "Suites"]
        name = random.choice(HOTEL_CHAINS + [f"The {city} {random.choice(suffixes)}"])
        hotels.append({
            "hotel_id": f"HTL{random.randint(10_000, 99_999)}",
            "name": name,
            "stars": stars,
            "address": f"{random.randint(1, 200)} {random.choice(['Main St', 'Park Ave', 'Central Blvd', 'Kings Rd', 'Grand Ave'])}, {city}",
            "price_per_night": ppn,
            "total_price": ppn * nights,
            "nights": nights,
            "guests": guests,
            "guest_rating": round(random.uniform(7.5, 9.8), 1),
            "review_count": random.randint(120, 8_000),
            "amenities": random.sample(amenity_pool, random.randint(4, 8)),
            "cancellation_policy": random.choice([
                "Free cancellation until 24h before check-in",
                "Free cancellation until 48h before check-in",
                "Free cancellation until 7 days before check-in",
                "Non-refundable (save 10%)",
            ]),
            "breakfast_included": random.choice([True, False]),
            "distance_to_centre": f"{random.uniform(0.1, 4.0):.1f} km from city centre",
        })

    hotels.sort(key=lambda x: x["guest_rating"], reverse=True)
    return {"city": city, "check_in": check_in, "check_out": check_out,
            "nights": nights, "hotels": hotels[:6]}


def _packing_tips(pattern: str) -> list[str]:
    return {
        "tropical":      ["Light breathable clothing", "SPF 50+ sunscreen", "Insect repellent",
                          "Compact rain jacket (afternoon showers)", "Flip flops + sandals"],
        "mediterranean": ["Light layers for evenings", "Comfortable walking shoes", "Sunglasses",
                          "Light cardigan or jacket", "Smart-casual for restaurants"],
        "continental":   ["Layered clothing", "Comfortable walking shoes", "Compact umbrella",
                          "Light sweater or hoodie", "Waterproof jacket"],
        "cold":          ["Heavy winter coat", "Thermal base layers", "Waterproof boots",
                          "Gloves, scarf & hat", "Thick wool socks"],
    }.get(pattern, ["Comfortable walking shoes", "Layers for changing weather", "Travel adapter"])


def _get_weather_forecast(args: dict) -> dict:
    city       = args["city"]
    start_date = args["start_date"]
    end_date   = args.get("end_date")

    pattern_key = CITY_WEATHER_PATTERN.get(city.split(",")[0].strip(), "continental")
    pattern     = WEATHER_PATTERNS[pattern_key]

    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        days  = max(1, (datetime.strptime(end_date, "%Y-%m-%d") - start).days + 1) if end_date else 7
    except ValueError:
        start = datetime.now()
        days  = 7

    forecast = []
    for i in range(min(days, 10)):
        d    = start + timedelta(days=i)
        high = random.randint(*pattern["temp_high"])
        low  = random.randint(*pattern["temp_low"])
        cond = random.choice(pattern["conditions"])
        rainy = any(kw in cond for kw in ("Rain", "Snow", "Thunder", "Freezing"))
        forecast.append({
            "date":             d.strftime("%Y-%m-%d (%a)"),
            "condition":        cond,
            "high_c":           high,
            "low_c":            low,
            "high_f":           int(high * 9 / 5 + 32),
            "low_f":            int(low  * 9 / 5 + 32),
            "precipitation_%":  random.randint(20, 75) if rainy else random.randint(0, 25),
            "humidity":         f"{random.randint(40, 85)}%",
            "wind_kmh":         random.randint(5, 35),
        })

    return {
        "city": city,
        "weather_pattern": pattern_key,
        "forecast": forecast,
        "packing_tips": _packing_tips(pattern_key),
    }


def _find_attractions(args: dict) -> dict:
    city       = args["city"]
    categories = args.get("categories") or ["museums", "historical", "restaurants", "nature"]
    budget     = args.get("budget", "moderate")

    city_key = city.split(",")[0].strip()
    city_data = ATTRACTION_DATA.get(city_key, {
        "museums":    [f"{city_key} National Museum", f"{city_key} Art Gallery"],
        "historical": [f"{city_key} Old Town", f"{city_key} Cathedral"],
        "restaurants":[f"Best {city_key} Cuisine", f"Local {city_key} Kitchen"],
        "shopping":   [f"{city_key} Main Market", f"{city_key} Shopping District"],
        "nightlife":  [f"{city_key} Jazz Club", f"Downtown {city_key} Bar"],
        "nature":     [f"{city_key} National Park", f"{city_key} Gardens"],
    })

    fee_range = {"budget": (0, 12), "moderate": (10, 35), "luxury": (40, 200)}

    results: dict[str, list[dict]] = {}
    for cat in categories:
        if cat not in city_data:
            continue
        results[cat] = []
        for name in city_data[cat]:
            lo, hi = fee_range[budget]
            fee    = random.randint(lo, hi)
            results[cat].append({
                "name":             name,
                "rating":           round(random.uniform(4.1, 4.9), 1),
                "reviews":          random.randint(500, 60_000),
                "entry_fee":        f"${fee}" if fee > 0 else "Free",
                "typical_duration": f"{random.randint(1, 3)} hours",
                "best_time":        random.choice(["Morning", "Afternoon", "Evening", "Anytime"]),
                "booking_required": random.choice([True, False]),
                "insider_tip":      random.choice([
                    "Book tickets online to skip the queue",
                    "Visit on weekdays to avoid weekend crowds",
                    "Free entry on the first Sunday of each month",
                    "Guided tours available in English",
                    "Arrive at opening time for the best experience",
                ]),
            })

    return {"city": city, "budget_level": budget, "attractions": results}


def _get_travel_requirements(args: dict) -> dict:
    from_c = args["from_country"].strip().upper()
    to_c   = args["to_country"].strip()

    # normalise common variations
    alias = {"USA": "US", "UNITED STATES": "US", "AMERICA": "US",
             "ENGLAND": "UK", "GREAT BRITAIN": "UK", "BRITAIN": "UK"}
    from_c_norm = alias.get(from_c, from_c)

    visa_info: dict | None = None
    for (fc, tc), info in VISA_DATA.items():
        if fc.upper() == from_c_norm and tc.upper() == to_c.upper():
            visa_info = info
            break

    if visa_info is None:
        visa_info = {
            "required": None,
            "notes": f"Visa requirements vary. Check the official {to_c.title()} embassy website or IATA Travel Centre for the latest information.",
        }

    return {
        "from_country":       from_c_norm,
        "to_country":         to_c.title(),
        "visa_required":      visa_info.get("required"),
        "visa_type":          visa_info.get("type", "N/A"),
        "visa_cost":          visa_info.get("cost", "N/A"),
        "notes":              visa_info["notes"],
        "passport_validity":  "Passport must be valid for at least 6 months beyond your return date",
        "recommended_vaccinations": random.choice([
            "No mandatory vaccines. COVID-19 up to date recommended.",
            "No mandatory vaccines. Hepatitis A & Typhoid recommended for rural travel.",
            "Yellow fever certificate required if arriving from endemic regions.",
        ]),
        "customs_allowances": {
            "alcohol":   "2 litres spirits or 4 litres wine",
            "tobacco":   "200 cigarettes or 50 cigars",
            "cash":      "Declare amounts over $10,000 USD equivalent",
        },
        "emergency_numbers": {
            "general":   "112 (EU) / 999 (UK) / 911 (US & Canada)",
            "your_embassy": f"Locate your country's embassy at travel.state.gov or equivalent",
        },
    }


def _get_currency_info(args: dict) -> dict:
    dest        = args["destination_country"].strip().title()
    home_ccy    = args.get("home_currency", "USD").upper()
    amount      = float(args.get("amount", 100))

    info = CURRENCY_DATA.get(dest, {
        "currency": "Local Currency",
        "code":     "LCY",
        "rate_to_usd": 1.0,
        "tips": ["Check rates before travelling", "Use ATMs for best rates", "Notify your bank"],
    })

    # rate: 1 home_ccy → dest_ccy  (simplified: assume home = USD)
    rate      = 1.0 / info["rate_to_usd"]
    converted = amount * rate

    daily = {
        "budget":   {"usd": 80,  "local": int(80  * rate)},
        "moderate": {"usd": 180, "local": int(180 * rate)},
        "luxury":   {"usd": 400, "local": int(400 * rate)},
    }

    return {
        "destination":      dest,
        "local_currency":   f"{info['currency']} ({info['code']})",
        "exchange_rate":    f"1 {home_ccy} ≈ {rate:.2f} {info['code']}",
        "conversion":       f"{amount:,.0f} {home_ccy} = {converted:,.0f} {info['code']}",
        "daily_budgets":    {
            "budget_traveller":  f"≈ {info['code']} {daily['budget']['local']:,}/day (${daily['budget']['usd']})",
            "mid_range":         f"≈ {info['code']} {daily['moderate']['local']:,}/day (${daily['moderate']['usd']})",
            "luxury":            f"≈ {info['code']} {daily['luxury']['local']:,}/day (${daily['luxury']['usd']})",
        },
        "money_tips":        info["tips"],
        "best_exchange_methods": [
            "Use local ATMs at your destination for competitive rates",
            "Avoid airport exchange booths (high fees and poor rates)",
            "Wise or Revolut cards offer near-interbank rates",
            "Notify your bank before travelling to avoid blocked transactions",
        ],
    }


def _create_itinerary(args: dict) -> dict:
    destination = args["destination"]
    start_date  = args["start_date"]
    end_date    = args["end_date"]
    interests   = args.get("interests") or ["culture", "food", "sightseeing"]
    pace        = args.get("pace", "moderate")

    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        days  = max(1, (datetime.strptime(end_date, "%Y-%m-%d") - start).days + 1)
    except ValueError:
        start = datetime.now()
        days  = 5

    acts_per_day = {"relaxed": 2, "moderate": 3, "packed": 5}[pace]

    city_key  = destination.split(",")[0].strip()
    city_data = ATTRACTION_DATA.get(city_key, {
        "historical":  [f"Old {city_key}", f"{city_key} Monument", f"Historic District"],
        "museums":     [f"{city_key} Museum", f"Local Heritage Centre"],
        "restaurants": [f"Local {city_key} Cuisine", f"{city_key} Food Market", f"Top-rated {city_key} Bistro"],
        "nature":      [f"{city_key} Park", f"{city_key} Gardens", f"Scenic Viewpoint"],
        "shopping":    [f"{city_key} Main Market", f"Artisan Quarter"],
    })

    # Collect activities matching traveller interests
    all_activities: list[tuple[str, str]] = []
    for cat, items in city_data.items():
        if any(interest.lower() in cat.lower() or cat.lower() in interest.lower()
               for interest in interests):
            all_activities.extend((cat, item) for item in items)
    if not all_activities:
        for cat, items in city_data.items():
            all_activities.extend((cat, item) for item in items)

    random.shuffle(all_activities)
    idx = 0

    time_slots = {
        2: ["10:00 AM", "7:00 PM"],
        3: ["9:30 AM", "2:00 PM", "7:30 PM"],
        5: ["9:00 AM", "11:00 AM", "1:30 PM", "4:00 PM", "7:30 PM"],
    }[acts_per_day]

    itinerary = []
    for day_n in range(min(days, 14)):
        date = start + timedelta(days=day_n)
        activities = []
        for slot in time_slots:
            cat, act = all_activities[idx % len(all_activities)]
            idx += 1
            activities.append({
                "time":     slot,
                "activity": act,
                "category": cat,
                "duration": "1–2 hours" if "restaurant" in cat.lower() else "2–3 hours",
                "tip":      random.choice([
                    "Pre-book online to avoid queues",
                    "Best light for photos in the morning",
                    "Allow extra time — it's worth it",
                    "Comfortable shoes recommended",
                    "Great for solo travellers and families alike",
                ]),
            })

        itinerary.append({
            "day":         day_n + 1,
            "date":        date.strftime("%Y-%m-%d, %A"),
            "theme":       random.choice(["History & Culture", "Food & Markets", "Art & Architecture",
                                          "Nature & Outdoors", "Local Life", "Relaxation"]),
            "activities":  activities,
            "daily_tip":   random.choice([
                f"Download offline maps of {city_key} before heading out",
                "Carry a reusable water bottle — stay hydrated",
                "Try the local street food today",
                "Public transport is cheap and reliable here",
                "A good day to splurge on a nice dinner",
            ]),
        })

    return {
        "destination": destination,
        "dates":       f"{start_date} to {end_date}",
        "duration":    f"{days} day{'s' if days > 1 else ''}",
        "pace":        pace,
        "interests":   interests,
        "itinerary":   itinerary,
        "general_tips": [
            "Book popular attractions at least 48h in advance",
            f"Learn a few basic phrases in the local language",
            "Get travel insurance — it's worth it",
            "Keep digital + physical copies of your passport and documents",
        ],
    }


def _estimate_trip_cost(args: dict) -> dict:
    destination   = args["destination"]
    duration      = int(args["duration_days"])
    travelers     = int(args.get("travelers", 1))
    budget_level  = args.get("budget_level", "moderate")
    incl_flights  = args.get("include_flights", True)

    city_key  = destination.split(",")[0].strip()
    daily     = DAILY_COST_USD.get(city_key, DEFAULT_DAILY_COST)[budget_level]

    flight_pp = {"budget": random.randint(250, 550),
                 "moderate": random.randint(550, 1_400),
                 "luxury": random.randint(2_500, 9_000)}[budget_level] if incl_flights else 0

    hotel_ppn     = int(daily * 0.38)
    food_pd       = int(daily * 0.28)
    activities_pd = int(daily * 0.20)
    transport_pd  = int(daily * 0.14)

    hotel_total  = hotel_ppn * duration
    food_total   = food_pd * duration
    act_total    = activities_pd * duration
    trans_total  = transport_pd * duration

    per_person   = flight_pp + hotel_total + food_total + act_total + trans_total
    group_total  = per_person * travelers
    contingency  = int(group_total * 0.10)

    return {
        "destination":    destination,
        "duration":       f"{duration} days",
        "travelers":      travelers,
        "budget_level":   budget_level,
        "per_person_breakdown": {
            "flights":           f"${flight_pp:,}"   if incl_flights else "Not included",
            "accommodation":     f"${hotel_total:,}  (${hotel_ppn:,}/night)",
            "food_&_drink":      f"${food_total:,}   (${food_pd:,}/day)",
            "activities":        f"${act_total:,}    (${activities_pd:,}/day)",
            "local_transport":   f"${trans_total:,}  (${transport_pd:,}/day)",
        },
        "subtotal_per_person":  f"${per_person:,}",
        "total_for_group":      f"${group_total:,}",
        "contingency_10pct":    f"${contingency:,}",
        "grand_total":          f"${group_total + contingency:,}",
        "money_saving_tips": [
            "Book flights 6–8 weeks ahead for the best fares",
            "Travel in shoulder season (spring/autumn) for 20–40% savings",
            "Use public transport instead of taxis",
            "Look for free museum days and city passes",
            "Stay slightly outside the tourist centre for cheaper accommodation",
        ],
    }


# ---------------------------------------------------------------------------
# MCP tool registry
# ---------------------------------------------------------------------------

TOOL_HANDLERS = {
    "search_flights":         _search_flights,
    "search_hotels":          _search_hotels,
    "get_weather_forecast":   _get_weather_forecast,
    "find_attractions":       _find_attractions,
    "get_travel_requirements":_get_travel_requirements,
    "get_currency_info":      _get_currency_info,
    "create_itinerary":       _create_itinerary,
    "estimate_trip_cost":     _estimate_trip_cost,
}


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="search_flights",
            description="Search for available flights between two cities. Returns multiple options with prices, airlines, times, and stops.",
            inputSchema={
                "type": "object",
                "properties": {
                    "origin":         {"type": "string", "description": "Departure city or IATA airport code (e.g. NYC, JFK, London)"},
                    "destination":    {"type": "string", "description": "Destination city or IATA airport code"},
                    "departure_date": {"type": "string", "description": "Departure date in YYYY-MM-DD format"},
                    "return_date":    {"type": "string", "description": "Return date for round trips (YYYY-MM-DD), optional"},
                    "passengers":     {"type": "integer", "description": "Number of passengers", "default": 1},
                    "cabin_class":    {"type": "string", "enum": ["economy", "business", "first"], "default": "economy"},
                },
                "required": ["origin", "destination", "departure_date"],
            },
        ),
        types.Tool(
            name="search_hotels",
            description="Search for available hotels at a destination. Returns options with prices, ratings, amenities, and cancellation policies.",
            inputSchema={
                "type": "object",
                "properties": {
                    "city":                 {"type": "string", "description": "City to search hotels in"},
                    "check_in":             {"type": "string", "description": "Check-in date (YYYY-MM-DD)"},
                    "check_out":            {"type": "string", "description": "Check-out date (YYYY-MM-DD)"},
                    "guests":               {"type": "integer", "description": "Number of guests", "default": 1},
                    "max_price_per_night":  {"type": "number",  "description": "Maximum price per night in USD (optional)"},
                    "star_rating":          {"type": "integer", "description": "Minimum star rating 1–5 (optional)"},
                },
                "required": ["city", "check_in", "check_out"],
            },
        ),
        types.Tool(
            name="get_weather_forecast",
            description="Get a day-by-day weather forecast for a destination during travel dates. Includes temperatures, conditions, and packing tips.",
            inputSchema={
                "type": "object",
                "properties": {
                    "city":       {"type": "string", "description": "City name"},
                    "start_date": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                    "end_date":   {"type": "string", "description": "End date (YYYY-MM-DD), optional — defaults to 7-day forecast"},
                },
                "required": ["city", "start_date"],
            },
        ),
        types.Tool(
            name="find_attractions",
            description="Find tourist attractions, restaurants, and points of interest at a destination, filtered by category and budget.",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "categories": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["museums", "historical", "restaurants", "shopping", "nightlife", "nature"]},
                        "description": "Categories of attractions to find",
                    },
                    "budget": {"type": "string", "enum": ["budget", "moderate", "luxury"], "default": "moderate"},
                },
                "required": ["city"],
            },
        ),
        types.Tool(
            name="get_travel_requirements",
            description="Get visa, passport, vaccination, and entry requirements for travelling between two countries.",
            inputSchema={
                "type": "object",
                "properties": {
                    "from_country": {"type": "string", "description": "Traveller's passport/citizenship country (e.g. US, UK, France)"},
                    "to_country":   {"type": "string", "description": "Destination country (e.g. Japan, Australia, Brazil)"},
                },
                "required": ["from_country", "to_country"],
            },
        ),
        types.Tool(
            name="get_currency_info",
            description="Get currency exchange rates, daily budget estimates, and money tips for a destination country.",
            inputSchema={
                "type": "object",
                "properties": {
                    "destination_country": {"type": "string", "description": "Destination country name"},
                    "home_currency":       {"type": "string", "description": "Your home currency code (e.g. USD, EUR, GBP)", "default": "USD"},
                    "amount":              {"type": "number",  "description": "Amount to convert for reference", "default": 100},
                },
                "required": ["destination_country"],
            },
        ),
        types.Tool(
            name="create_itinerary",
            description="Create a detailed day-by-day travel itinerary for a destination, tailored to interests and travel pace.",
            inputSchema={
                "type": "object",
                "properties": {
                    "destination": {"type": "string", "description": "Destination city or region"},
                    "start_date":  {"type": "string", "description": "Trip start date (YYYY-MM-DD)"},
                    "end_date":    {"type": "string", "description": "Trip end date (YYYY-MM-DD)"},
                    "interests": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Traveller interests, e.g. ['history', 'food', 'art', 'nature', 'shopping', 'nightlife']",
                    },
                    "pace": {"type": "string", "enum": ["relaxed", "moderate", "packed"], "default": "moderate",
                             "description": "relaxed = 2 activities/day, moderate = 3, packed = 5"},
                },
                "required": ["destination", "start_date", "end_date"],
            },
        ),
        types.Tool(
            name="estimate_trip_cost",
            description="Estimate the total cost of a trip broken down by flights, accommodation, food, activities, and transport.",
            inputSchema={
                "type": "object",
                "properties": {
                    "destination":    {"type": "string",  "description": "Destination city"},
                    "duration_days":  {"type": "integer", "description": "Number of days"},
                    "travelers":      {"type": "integer", "description": "Number of travellers", "default": 1},
                    "budget_level":   {"type": "string",  "enum": ["budget", "moderate", "luxury"], "default": "moderate"},
                    "include_flights":{"type": "boolean", "description": "Whether to include flight cost estimate", "default": True},
                },
                "required": ["destination", "duration_days"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        result = {"error": f"Unknown tool: {name}"}
    else:
        try:
            result = handler(arguments)
        except Exception as exc:
            result = {"error": str(exc)}

    return [types.TextContent(type="text", text=json.dumps(result, indent=2))]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
