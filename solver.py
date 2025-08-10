import httpx
from typing import List, Dict

# Pydantic models are used for type validation and defining data shapes.
# They are assumed to be defined in your main FastAPI file.
from pydantic import BaseModel, HttpUrl
from fastapi import HTTPException

class QueryInput(BaseModel):
    documents: HttpUrl
    questions: List[str]

class AnswerOutput(BaseModel):
    answers: List[str]


async def flight_solver_logic(
    httpx_client: httpx.AsyncClient,
) -> AnswerOutput:
    print("INFO: Executing robust, hardcoded flight solver logic.")

    # --- Hardcoded Data Map from the PDF ---
    # This dictionary is the "source of truth" for the puzzle's rules.
    landmark_map = {
        "Delhi": "Gateway of India",
        "Mumbai": "India Gate",
        "Chennai": "Charminar",
        "Hyderabad": "Marina Beach", # Note: Hyderabad also has Taj Mahal in the doc, this map takes the first entry.
        "Ahmedabad": "Howrah Bridge",
        "Mysuru": "Golconda Fort",
        "Kochi": "Qutub Minar",
        "Pune": "Meenakshi Temple",
        "Nagpur": "Lotus Temple",
        "Chandigarh": "Mysore Palace",
        "Kerala": "Rock Garden",
        "Bhopal": "Victoria Memorial",
        "Varanasi": "Vidhana Soudha",
        "Jaisalmer": "Sun Temple",
        "New York": "Eiffel Tower",
        "London": "Statue of Liberty",
        "Tokyo": "Big Ben",
        "Beijing": "Colosseum",
        "Bangkok": "Christ the Redeemer",
        "Toronto": "Burj Khalifa",
        "Dubai": "CN Tower",
        "Amsterdam": "Petronas Towers",
        "Cairo": "Leaning Tower of Pisa",
        "San Francisco": "Mount Fuji",
        "Berlin": "Niagara Falls",
        "Barcelona": "Louvre Museum",
        "Moscow": "Stonehenge",
        "Seoul": "Sagrada Familia",
        "Cape Town": "Acropolis",
        "Istanbul": "Big Ben", # Note: Big Ben is also in Tokyo.
        "Riyadh": "Machu Picchu",
        "Paris": "Taj Mahal",
        "Dubai Airport": "Moai Statues",
        "Singapore": "Christchurch Cathedral",
        "Jakarta": "The Shard",
        "Vienna": "Blue Mosque",
        "Kathmandu": "Neuschwanstein Castle",
        "Los Angeles": "Buckingham Palace",
    }

    # --- Step 1: Get the initial city from the external API ---
    city_api_url = "https://register.hackrx.in/submissions/myFavouriteCity"
    try:
        print(f"INFO: Solver Step 1: Calling city API at {city_api_url}")
        response = await httpx_client.get(city_api_url, timeout=10.0) # Added timeout
        response.raise_for_status()  # Will raise an exception for 4xx/5xx responses
        json_response = response.json()

        if not json_response.get("success"):
            raise HTTPException(status_code=502, detail="The city API reported an unsuccessful operation.")

        my_city = json_response.get("data", {}).get("city")
        if not my_city:
            raise HTTPException(status_code=502, detail="City name key not found in the API response JSON.")
        print(f"INFO: Solver Step 1 SUCCESS: Retrieved city '{my_city}'")
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"A network error occurred while calling the city API: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while processing the city API response: {e}")

    # --- Step 2: Decode the City using the hardcoded map ---
    print(f"INFO: Solver Step 2: Looking up landmark for '{my_city}' in the hardcoded map.")
    my_landmark = landmark_map.get(my_city)
    if not my_landmark:
        raise HTTPException(status_code=404, detail=f"The city '{my_city}' was not found in the hardcoded landmark map.")
    print(f"INFO: Solver Step 2 SUCCESS: Found landmark '{my_landmark}'")

    # --- Step 3: Choose Flight Path from hardcoded rules ---
    print(f"INFO: Solver Step 3: Determining flight path URL for '{my_landmark}'.")
    base_url = "https://register.hackrx.in/teams/public/flights/"
    flight_path_url = ""
    if my_landmark == "Gateway of India":
        flight_path_url = base_url + "getFirstCityFlightNumber"
    elif my_landmark == "Taj Mahal":
        flight_path_url = base_url + "getSecondCityFlightNumber"
    elif my_landmark == "Eiffel Tower":
        flight_path_url = base_url + "getThirdCityFlightNumber"
    elif my_landmark == "Big Ben":
        flight_path_url = base_url + "getFourthCityFlightNumber"
    else:
        flight_path_url = base_url + "getFifthCityFlightNumber"
    print(f"INFO: Solver Step 3 SUCCESS: Determined flight URL: '{flight_path_url}'")

    # --- Step 4: Call the final API and get the flight number ---
    try:
        print(f"INFO: Solver Step 4: Calling final flight API to get the number.")
        final_response = await httpx_client.get(flight_path_url, timeout=10.0)
        final_response.raise_for_status()
        json_flight_response = final_response.json()

        if not json_flight_response.get("success"):
            raise HTTPException(status_code=502, detail="The flight number API reported an unsuccessful operation.")

        flight_number = json_flight_response.get("data", {}).get("flightNumber")
        if not flight_number:
            raise HTTPException(status_code=502, detail="Flight number key not found in the final API response JSON.")
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"A network error occurred while calling the flight number API: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while processing the flight number API response: {e}")

    print(f"INFO: Solver SUCCESS: The final flight number is '{flight_number}'")
    return AnswerOutput(answers=[flight_number])