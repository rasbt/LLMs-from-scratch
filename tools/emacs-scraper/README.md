
Emacs Scraper (text/images/links)


Purpose
- Scrape web pages into plain text, optionally including image URLs and hyperlinks, and save as .md files under output/.
- Automatically numbers outputs using the highest existing suffix + 1.

Install
1) Add this file to your load-path:
   (add-to-list 'load-path "/workspaces/LLMs-from-scratch/tools/emacs-scraper")
2) Require it:
   (require 'emacs-scraper)

Usage
- M-x emacs-scraper-open
- M-x emacs-scraper-fetch-url
- M-x emacs-scraper-fetch-urls

Keybindings (enable minor mode)
- M-x emacs-scraper-mode
- C-c s o : open UI
- C-c s u : fetch URL
- C-c s m : fetch multiple URLs


Behavior
- Saves output to output/ as .md files. Output can include:
   - Main text (always)
   - Image URLs (if enabled)
   - Hyperlinks (if enabled)
- Filename format: <base-name>-<next-number>.md
- Base name defaults to a slugified URL.


Options
- To include images, set `emacs-scraper-save-images` to t (or check the box in the UI).
- To include links, set `emacs-scraper-save-links` to t (or check the box in the UI).

Notes
- The number always increments to max+1 even if gaps exist.
