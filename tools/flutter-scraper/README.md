# Flutter Scraper

A simple Flutter app to scrape a web page and save text, image URLs, and hyperlinks to a numbered markdown file in the output/ directory.

## Features
- Extracts main text, image URLs, and hyperlinks from a given URL
- Saves output as output/<base-name>-<next-number>.md (number always increments to max+1)
- Text-only, or optionally includes images/links

## Usage

1. Create the Flutter project in this directory if not already present:
   ```bash
   flutter create .
   ```
2. Add dependencies to pubspec.yaml:
   ```yaml
   dependencies:
     http: ^1.0.0
     html: ^0.15.0
     path: ^1.8.0
   ```
3. Run the app (see main.dart for CLI or UI usage):
   ```bash
   flutter run -d <device> # or use flutter pub run for CLI
   ```

## Example
- See main.dart for a simple CLI/console implementation.
- Output will be saved in output/ as a markdown file.
