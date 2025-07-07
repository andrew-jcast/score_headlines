import requests
from bs4 import BeautifulSoup
from typing import List

def fetch_nyt_headlines() -> List[str]:
    """Fetch headlines from New York Times homepage."""
    print("Fetching headlines from New York Times")
    
    try:
        response = requests.get('https://www.nytimes.com/', timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching NYT headlines, requests related: {e}")
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    headlines = []
    
    for element in soup.find_all(class_="story-wrapper"):
        paragraphs = element.find_all('p')
        
        if not paragraphs:
            continue
            
        if len(paragraphs) == 1:
            headline_text = paragraphs[0].text.strip()
        elif len(paragraphs) >= 2:
            # Skip category tags (first paragraph if it's very short)
            if len(paragraphs[0].text) < 20:
                title = paragraphs[1].text.strip() if len(paragraphs) > 1 else ""
                summary = paragraphs[2].text.strip() if len(paragraphs) > 2 else ""
            else:
                title = paragraphs[0].text.strip()
                summary = paragraphs[1].text.strip()
            
            headline_text = f"{title}.{summary}" if summary else title
        else:
            continue
            
        if headline_text:
            headlines.append(headline_text)
    
    print(f"Fetched {len(headlines)} NYT headlines")
    return headlines


def fetch_chicago_tribune_headlines() -> List[str]:
    """Fetch headlines from Chicago Tribune homepage."""
    print("Fetching headlines from Chicago Tribune...")
    
    try:
        response = requests.get('https://www.chicagotribune.com/', timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching Chicago Tribune headlines: {e}")
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    headlines = []
    
    for element in soup.find_all(class_="article-title"):
        if 'title' in element.attrs:
            headline_text = element['title'].strip()
            if headline_text and len(headline_text) > 5:
                headlines.append(headline_text)
    
    print(f"Found {len(headlines)} Chicago Tribune headlines")
    return headlines