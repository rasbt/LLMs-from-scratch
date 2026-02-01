# Python Scraper

A simple Python script to scrape a web page and save text, image URLs, and hyperlinks to a numbered markdown file in the output/ directory.

## Features
- Extracts main text, image URLs, and hyperlinks from a given URL
- Saves output as output/<base-name>-<next-number>.md (number always increments to max+1)
- Text-only, or optionally includes images/links

## Usage

1. Install requirements:
   ```bash
   pip install requests beautifulsoup4
   ```
2. Run the scraper:
   ```bash
   python scraper.py <url> [--images] [--links]
   ```
   - `--images`: include image URLs
   - `--links`: include hyperlinks

## Example
```bash
python scraper.py https://docs.flutter.dev/ --images --links
```

This will save a markdown file in output/ with the extracted content.
