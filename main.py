from openai import OpenAI
import json
import requests
from dotenv import dotenv_values

env = dotenv_values()
client = OpenAI(api_key=env['OPENAI_API_KEY'])

params = {
    "key": env['AMAP_API_KEY'],
    "output": "json",
    "extensions": "all",
}

# https://lbs.amap.com/api/webservice/guide/api/weatherinfo
def query_city_weather(city):
    params["city"] = city

    with requests.Session() as session:
        response = session.get("https://restapi.amap.com/v3/weather/weatherInfo", params=params)
        weather_data = response.json()

    return json.dumps(weather_data)

def get_current_weather(location, unit="celsius"):
    """Get the current weather in a given location"""
    if "shanghai" in location.lower():
        return query_city_weather("上海")
    elif "beijing" in location.lower():
        return query_city_weather("北京")
    elif "sanya" in location.lower():
        return query_city_weather("三亚")
    else:
        return json.dumps({"location": location, "temperature": "unknown"})

def run_conversation():
    # Step 1: send the conversation and available functions to the model
    messages = [
        {"role": "user", "content": "What's the weather like in shanghai, beijing, and sanya? Please express the result in Chinese."}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=messages,
        tools=tools,
        tool_choice="auto",  # auto is default, but we'll be explicit
    )
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    # Step 2: check if the model wanted to call a function
    if tool_calls:
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "get_current_weather": get_current_weather,
        }  # only one function in this example, but you can have multiple
        # extend conversation with assistant's reply
        messages.append(response_message)
        # Step 4: send the info for each function call and function response to the model
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(
                location=function_args.get("location"),
                unit=function_args.get("unit"),
            )
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )  # extend conversation with function response
        second_response = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=messages,
        )  # get a new response from the model where it can see the function response
        return second_response

print(run_conversation())
