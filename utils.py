#Dependent functions for news scrapper
from urllib.parse import quote_plus
from dotenv import load_dotenv
import requests
import os
from fastapi import FastAPI, HTTPException
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from datetime import datetime
from elevenlabs import ElevenLabs


# Create a URL against the topic a user is sending from the frontend
# A function that given a topic returns a URL
def generate_valid_news_url(keyword: str) -> str:
    """
    Generate a Google News search URL for a keyword with optional sorting by latest
    
    Args:
        keyword: Search term to use in the news search
        
    Returns:
        str: Constructed Google News search URL
    """
    q = quote_plus(keyword)
    return f"https://news.google.com/search?q={q}&tbs=sbd:1"


#Scrape the url using Brightdata
def scrape_with_brightdata(url: str) -> str:
    #Returns the HTML returned from the url
    """Scrape a URL using BrightData"""
    headers = {
        "Authorization": f"Bearer {os.getenv('BRIGHTDATA_API_TOKEN')}",
        "Content-Type": "application/json"
    }
    payload = {
        "zone": os.getenv('BRIGHTDATA_WEB_UNLOCKER_ZONE'),
        "url": url,
        "format": "raw"
    }
    try:
        response = requests.post("https://api.brightdata.com/request", json=payload, headers=headers)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"BrightData error: {str(e)}")
    
#Clean the HTML to text using BeautifulSoup
def clean_html_to_text(html_content: str) -> str:
    """Clean HTML content to plain text"""
    soup = BeautifulSoup(html_content, "html.parser")
    text = soup.get_text(separator="\n")
    return text.strip()


#Extract the headlines from the text and only the important information
def extract_headlines(cleaned_text: str) -> str:
    """
    Extract and concatenate headlines from cleaned news text content.
    Args:
        cleaned_text: Raw text from news page after HTML cleaning
    Returns:
        str: Combined headlines separated by newlines
    """
    headlines = []
    current_block = []
    # Split text into lines and remove empty lines
    lines = [line.strip() for line in cleaned_text.split('\n') if line.strip()]
    # Process lines to find headline blocks
    for line in lines:
        if line == "More":
            if current_block:
                # First line of block is headline
                headlines.append(current_block[0])
                current_block = []
        else:
            current_block.append(line)
    # Add any remaining block at end of text
    if current_block:
        headlines.append(current_block[0])
    return "\n".join(headlines)

def summarize_with_openai_news_script(api_key: str, headlines: str) -> str:
    """
    Summarize multiple news headlines into a TTS-friendly broadcast news script using OpenAI model via LangChain.
    """
    system_prompt = """
You are my personal news editor and scriptwriter for a news podcast. Your job is to turn raw headlines into a clean, professional, and TTS-friendly news script.

The final output will be read aloud by a news anchor or text-to-speech engine. So:
- Do not include any special characters, emojis, formatting symbols, or markdown.
- Do not add any preamble or framing like "Here's your summary" or "Let me explain".
- Write in full, clear, spoken-language paragraphs.
- Keep the tone formal, professional, and broadcast-style — just like a real TV news script.
- Focus on the most important headlines and turn them into short, informative news segments that sound natural when spoken.
- Start right away with the actual script, using transitions between topics if needed.

Remember: Your only output should be a clean script that is ready to be read out loud.
"""

    try:
        llm = ChatOpenAI(
            model="gpt-3.5-turbo", 
            api_key=api_key,
            temperature=0.4,
            max_tokens=128
        )

        # Invoke Claude with system + user prompt
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=headlines)
        ])

        return response.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI error: {str(e)}")

# If there's more than one keyword, we need to generate a URL for each keyword and we can use this function to do that
def generate_news_urls_to_scrape(list_of_keywords):
    valid_urls_dict = {}
    for keyword in list_of_keywords:
        valid_urls_dict[keyword] = generate_valid_news_url(keyword)
    
    return valid_urls_dict


def generate_broadcast_news(api_key, news_data, reddit_data, topics):
    # Updated system message with flexible source handling
    system_prompt = """
    You are broadcast_news_writer, a professional virtual news reporter. Generate natural, TTS-ready news reports using available sources:

    For each topic, STRUCTURE BASED ON AVAILABLE DATA:
    1. If news exists: "According to official reports..." + summary
    2. If Reddit exists: "Online discussions on Reddit reveal..." + summary
    3. If both exist: Present news first, then Reddit reactions
    4. If neither exists: Skip the topic (shouldn't happen)

    Formatting rules:
    - ALWAYS start directly with the content, NO INTRODUCTIONS
    - Keep audio length 60-120 seconds per topic
    - Use natural speech transitions like "Meanwhile, online discussions..." 
    - Incorporate 1-2 short quotes from Reddit when available
    - Maintain neutral tone but highlight key sentiments
    - End with "To wrap up this segment..." summary

    Write in full paragraphs optimized for speech synthesis. Avoid markdown.
    """

    try:
        topic_blocks = []
        for topic in topics:
            news_content = news_data["news_analysis"].get(topic ) if news_data else ''
            reddit_content = reddit_data["reddit_analysis"].get(topic) if reddit_data else ''
            context = []
            if news_content:
                context.append(f"OFFICIAL NEWS CONTENT:\n{news_content}")
            if reddit_content:
                context.append(f"REDDIT DISCUSSION CONTENT:\n{reddit_content}")
            
            if context:  # Only include topics with actual content
                topic_blocks.append(
                    f"TOPIC: {topic}\n\n" +
                    "\n\n".join(context)
                )

        user_prompt = (
            "Create broadcast segments for these topics using available sources:\n\n" +
            "\n\n--- NEW TOPIC ---\n\n".join(topic_blocks)
        )

        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            api_key=api_key,
            temperature=0.3,
            max_tokens=128,
        )

        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])

        return response.content

    except Exception as e:
        raise e
    

def text_to_audio_elevenlabs_sdk(
    text: str,
    voice_id: str = "JBFqnCBsd6RMkjVDRZzb",
    model_id: str = "eleven_multilingual_v2",
    output_format: str = "mp3_44100_128",
    output_dir: str = "audio",
    api_key: str = None
) -> str:
    """
    Converts text to speech using ElevenLabs SDK and saves it to audio/ directory.

    Returns:
        str: Path to the saved audio file.
    """
    try:
        api_key = api_key or os.getenv("ELEVEN_API_KEY")
        if not api_key:
            raise ValueError("ElevenLabs API key is required.")

        # Initialize client
        client = ElevenLabs(api_key=api_key)

        # Get the audio generator
        audio_stream = client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id=model_id,
            output_format=output_format
        )

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Generate unique filename
        filename = f"tts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
        filepath = os.path.join(output_dir, filename)

        # Write audio chunks to file
        with open(filepath, "wb") as f:
            for chunk in audio_stream:
                f.write(chunk)

        return filepath

    except Exception as e:
        raise e
