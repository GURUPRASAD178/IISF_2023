import requests
from bs4 import BeautifulSoup
import os

# List of Bhuvan URLs (shortened here for demo - replace with full list in production)
bhuvan_urls = [
    "https://bhuvan.nrsc.gov.in",
    "https://bhuvan-app3.nrsc.gov.in/aadhaar/",
    "https://bhuvan-app2.nrsc.gov.in/mgnrega/mgnrega_phase2.php"
    "https://bhuvan.nrsc.gov.in",
    "https://bhuvan-app3.nrsc.gov.in/aadhaar/",
    "https://bhuvan-app2.nrsc.gov.in/mgnrega/mgnrega_phase2.php",
    "https://bhuvan-app3.nrsc.gov.in/data/",
    "https://bhuvan-app1.nrsc.gov.in/bhuvan2d/bhuvan/bhuvan2d.php",
    "https://bhuvan.nrsc.gov.in/home/index.php",
    "https://bhuvan-app1.nrsc.gov.in/api/",
    "https://bhuvan-app1.nrsc.gov.in/hfa/housing_for_all.php",
    "https://bhuvan-app1.nrsc.gov.in/apshcl",
    "https://bhuvan-app1.nrsc.gov.in/ntr",
    "https://bhuvan.nrsc.gov.in/forum",
    "https://bhuvan-wbis.nrsc.gov.in/",
    "https://bhuvan.nrsc.gov.in/geonetwork/",
    "https://bhuvan-app2.nrsc.gov.in/planner/",
    "https://bhuvan-app1.nrsc.gov.in/globe/3d.php",
    "https://bhuvan-app1.nrsc.gov.in/mhrd_rusa/",
    "https://bhuvan-app1.nrsc.gov.in/geographicalindication/index.php",

]

# Output folder
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

def extract_visible_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for script in soup(["script", "style", "noscript"]):
        script.decompose()
    return soup.get_text(separator=" ", strip=True)

def scrape_bhuvan():
    for url in bhuvan_urls:
        try:
            print(f"Scraping: {url}")
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                text = extract_visible_text(response.text)
                filename = url.replace("https://", "").replace("http://", "").replace("/", "_") + ".txt"
                with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
                    f.write(text)
            else:
                print(f"Failed to fetch: {url} (status: {response.status_code})")
        except Exception as e:
            print(f"Error fetching {url}: {e}")

if __name__ == "__main__":
    scrape_bhuvan()