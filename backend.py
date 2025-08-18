from fastapi import FastAPI, HTTPException, File, Response
from models import NewsRequest
from news_scrapper import NewsScraper
from dotenv import load_dotenv

app = FastAPI()
load_dotenv()


@app.post("/generate-news-audio")
async def generate_news_audio(request: NewsRequest):
    try:
        results = {}

        #Scrape data
        if request.source_type in ["news", "both"]:
            #scrape news
            news_scraper = NewsScraper()
            results["news"] = await news_scraper.scrape_news(request.topics)

        if request.source_type in ["reddit", "both"]:
            #scrape reddit
            results["reddit"] = {"reddit_scraped": "This is from reddit"}

        news_data = results.get("news", {})
        reddit_data = results.get("reddit", {})

        #Generate summary
        news_summary = my_summary_function(news_data, reddit_data)

        audio_path = convert_to_audio(news_summary)
        if audio_path:
            return response, headers, etc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend:app",
        host="0.0.0.0",
        port=1234,
        reload=True
    )