from langchain_community.tools import WikipediaQueryRun , DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import tool
from datetime import datetime

def save_to_txt(data:str,filename:str = "research_output.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"---Reasearch Output---\nTimestamp: {timestamp}\n\n{data}\n\n"
    with open(filename, "a" , encoding="utf-8") as file:
        file.write(formatted_text)
    return f"Data saved to {filename} at {timestamp}"

save_tool = Tool(
    name="save_txt_to_file",
    func=save_to_txt,
    description="Saves the structured research output to a text file with a timestamp.",
)

search = DuckDuckGoSearchRun()
search_tool = Tool(
    name = "search",
    func=search.run,
    description="Search the web for information.",
)

api_wrapper = WikipediaAPIWrapper(top_k_results=1 , doc_content_chars_max=100)
wikipedia_tool = WikipediaAPIWrapper(api_wrapper=api_wrapper)
 
