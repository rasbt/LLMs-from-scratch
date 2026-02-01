import os
import re
import argparse
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

def slugify(url):
    s = url.lower()
    s = re.sub(r"https?://", "", s)
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"^-+", "", s)
    s = re.sub(r"-+$", "", s)
    return s or "scrape"

def get_next_number(output_dir, base_name):
    maxn = 0
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    for f in os.listdir(output_dir):
        m = re.match(rf"{re.escape(base_name)}-(\d+)\.md$", f)
        if m:
            maxn = max(maxn, int(m.group(1)))
    return maxn + 1

def extract_content(url, include_images=False, include_links=False):
    resp = requests.get(url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    # Remove scripts/styles
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    out = text.strip()
    if include_images:
        images = [img.get("src") for img in soup.find_all("img") if img.get("src")]
        if images:
            out += "\n\n## Images\n" + "\n".join(images)
    if include_links:
        links = [a.get("href") for a in soup.find_all("a") if a.get("href")]
        if links:
            out += "\n\n## Links\n" + "\n".join(links)
    return out

def main():
    parser = argparse.ArgumentParser(description="Scrape a web page to markdown.")
    parser.add_argument("url", help="URL to scrape")
    parser.add_argument("--images", action="store_true", help="Include image URLs")
    parser.add_argument("--links", action="store_true", help="Include hyperlinks")
    args = parser.parse_args()

    output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "output")
    base_name = slugify(args.url)
    num = get_next_number(output_dir, base_name)
    filename = f"{base_name}-{num}.md"
    path = os.path.join(output_dir, filename)
    content = extract_content(args.url, args.images, args.links)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
        if not content.endswith("\n"):
            f.write("\n")
    print(f"Saved: {path}")

if __name__ == "__main__":
    main()
