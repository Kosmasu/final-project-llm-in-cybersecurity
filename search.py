from googlesearch import search
from openai import BaseModel
import requests
from bs4 import BeautifulSoup
from newspaper import Article


class SearchResult(BaseModel):
    url: str
    content: str

    def format(self) -> str:
        return f"""
URL: {self.url}
Content: 
```
{self.content[:1000]}...
```
        """.strip()


def google_search(query, num_results=10) -> list[str]:
    search_results = []
    try:
        for result in search(query, num_results=num_results):
            search_results.append(result)
        return search_results
    except Exception as e:
        print(f"An error occurred during the search: {e}")
        return []


def fetch_url_content(url) -> str | None:
    try:
        # Use newspaper3k to download and parse the article
        article = Article(url, fetch_images=False)
        article.download()
        article.parse()
        return str(article.text)
    except Exception as e:
        print(f"Error processing URL {url} with newspaper3k: {e}")
        # Fallback to BeautifulSoup if newspaper3k fails
        print(f"Falling back to BeautifulSoup for URL: {url}")
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            text = soup.get_text(separator=" ", strip=True)
            return text
        except requests.exceptions.RequestException as req_e:
            print(f"Error fetching URL {url} with requests: {req_e}")
            return None
        except Exception as bs_e:
            print(f"Error processing URL {url} with BeautifulSoup: {bs_e}")
            return None


def search_and_fetch_contents(query, num_results=3) -> list[SearchResult]:
    urls: list[str] = google_search(query, num_results=10)
    if not urls:
        print("No URLs found.")
        return []

    search_results: list[SearchResult] = []

    for i, url in enumerate(urls, 1):
        print(f"Processing URL {i}/{len(urls)}: {url}")
        content = fetch_url_content(url)
        if content:
            search_results.append(
                SearchResult(
                    url=url,
                    content=content,
                )
            )
        if len(search_results) >= num_results:
            break

    return search_results


# Example usage
if __name__ == "__main__":
    query = "what is phishing?"
    results = search_and_fetch_contents(query)
    print(results)
