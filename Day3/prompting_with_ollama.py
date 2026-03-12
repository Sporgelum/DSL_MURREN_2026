# %%
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
# %%
#!ollama pull gpt-oss:20b
!ollama list
# %%
# Initialize the Ollama LLM with the desired model
ollama_llm = OllamaLLM(model="gpt-oss:20b")
# %%
# Create a chat prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that provides information about the weather."),
    ("human", "What is the weather like in Mürren today?"),
])

# Format the prompt into a string
formatted_prompt = prompt.format()

response = ollama_llm.invoke(formatted_prompt)
print(response)

# %% Ollama native
from langchain_ollama import ChatOllama

llm = ChatOllama(model="gpt-oss:20b")

response = llm.invoke(prompt)
print(response)

# %%
import requests
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# 1. Fetch weather data from an API
url = "https://api.open-meteo.com/v1/forecast"
params = {
    "latitude": 46.56,   # Mürren area
    "longitude": 7.89,
    "hourly": "temperature_2m,precipitation",
    "forecast_days": 1
}
weather_data = requests.get(url, params=params).json()

# 2. Prepare the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a meteorology assistant. Analyze the provided weather data."),
    ("human", f"Here is the weather data: {weather_data}\n\nGive me the best forecast summary.")
])

# 3. Run Ollama
llm = OllamaLLM(model="gpt-oss:20b")
formatted = prompt.format()
response = llm.invoke(formatted)

print(response)


# %%
