;;; emacs-scraper.el --- Simple web scraper (text, images, links) -*- lexical-binding: t; -*-

;; Usage:
;;  - M-x emacs-scraper-open
;;  - M-x emacs-scraper-fetch-url
;;
;; Options:
;;  - Set `emacs-scraper-save-images` and/or `emacs-scraper-save-links` to t to include images/links in output.
(defcustom emacs-scraper-save-images nil
  "If non-nil, save image URLs found in the page."
  :type 'boolean
  :group 'emacs-scraper)

(defcustom emacs-scraper-save-links nil
  "If non-nil, save hyperlinks found in the page."
  :type 'boolean
  :group 'emacs-scraper)

(require 'url)
(require 'url-http)
(require 'shr)
(require 'subr-x)
(require 'widget)
(require 'wid-edit)

(defgroup emacs-scraper nil
  "Text-only web scraper for Emacs."
  :group 'tools)

(defcustom emacs-scraper-output-dir "output"
  "Relative output directory within the project root."
  :type 'string
  :group 'emacs-scraper)

(defcustom emacs-scraper-open-after-save t
  "Whether to open the saved file after scraping."
  :type 'boolean
  :group 'emacs-scraper)

(defvar emacs-scraper--buffer-name "*Emacs Scraper*")

(defun emacs-scraper--project-root ()
  "Return project root by locating .git or fallback to `default-directory`."
  (or (locate-dominating-file default-directory ".git")
      (locate-dominating-file default-directory emacs-scraper-output-dir)
      default-directory))

(defun emacs-scraper--output-dir ()
  "Return absolute output directory path."
  (expand-file-name emacs-scraper-output-dir (emacs-scraper--project-root)))

(defun emacs-scraper--ensure-output-dir ()
  "Ensure output directory exists."
  (let ((dir (emacs-scraper--output-dir)))
    (unless (file-directory-p dir)
      (make-directory dir t))
    dir))

(defun emacs-scraper--max-existing-number ()
  "Return the max numeric suffix among *.md files in output dir."
  (let* ((dir (emacs-scraper--output-dir))
         (files (when (file-directory-p dir)
                  (directory-files dir nil "\.md\'")))
         (maxn 0))
    (dolist (f files maxn)
      (when (string-match "-\([0-9]+\)\.md\'" f)
        (setq maxn (max maxn (string-to-number (match-string 1 f))))))))

(defun emacs-scraper--next-number ()
  "Return next numeric suffix (max + 1)."
  (1+ (emacs-scraper--max-existing-number)))

(defun emacs-scraper--slugify (s)
  "Convert string S to a safe filename slug."
  (let* ((s (downcase s))
         (s (replace-regexp-in-string "https?://" "" s))
         (s (replace-regexp-in-string "[^a-z0-9]+" "-" s))
         (s (replace-regexp-in-string "^-+" "" s))
         (s (replace-regexp-in-string "-+$" "" s)))
    (if (string-empty-p s) "scrape" s)))

(defun emacs-scraper--default-base-name (url)
  "Default base name from URL."
  (emacs-scraper--slugify url))

(defun emacs-scraper--target-path (base-name)
  "Return full output path with numeric suffix.
BASE-NAME should not include extension or suffix." 
  (let* ((dir (emacs-scraper--ensure-output-dir))
         (num (emacs-scraper--next-number))
         (file (format "%s-%d.md" base-name num)))
    (expand-file-name file dir)))


;; Helper to extract images and links from HTML DOM
(defun emacs-scraper--extract-images-and-links (dom)
  "Return cons of (image-urls . links) from DOM."
  (let ((imgs nil)
        (lnks nil))
    (cl-labels ((walk (node)
                  (when (consp node)
                    (let ((tag (car node)) (attrs (cadr node)))
                      (when (and (eq tag 'img) (listp attrs))
                        (let ((src (cdr (assoc 'src attrs))))
                          (when src (push src imgs))))
                      (when (and (eq tag 'a) (listp attrs))
                        (let ((href (cdr (assoc 'href attrs))))
                          (when href (push href lnks))))
                      (mapc #'walk (cddr node))))))
      (walk dom))
    (cons (nreverse imgs) (nreverse lnks))))

(defun emacs-scraper--extract-content (content-type)
  "Extract text, images, and links from current buffer body.
CONTENT-TYPE is a string like text/html."
  (let ((body (if (boundp 'url-http-end-of-headers)
                  url-http-end-of-headers
                (save-excursion
                  (goto-char (point-min))
                  (search-forward "\n\n" nil t)))))
    (goto-char (or body (point-min)))
    (cond
     ((and content-type (string-match-p "text/html" content-type))
      (let* ((dom (libxml-parse-html-region (point) (point-max)))
             (text (with-temp-buffer
                     (shr-insert-document dom)
                     (string-trim (buffer-string))))
             (imgs-lnks (emacs-scraper--extract-images-and-links dom))
             (imgs (car imgs-lnks))
             (lnks (cdr imgs-lnks)))
        (concat
         text
         (when (and emacs-scraper-save-images imgs)
           (concat "\n\n## Images\n" (mapconcat #'identity imgs "\n")))
         (when (and emacs-scraper-save-links lnks)
           (concat "\n\n## Links\n" (mapconcat #'identity lnks "\n"))))))
     (t
      (string-trim (buffer-substring-no-properties (point) (point-max))))))

(defun emacs-scraper--content-type ()
  "Return Content-Type header if available." 
  (or (and (boundp 'url-http-content-type) url-http-content-type)
      (save-excursion
        (goto-char (point-min))
        (when (re-search-forward "^Content-Type: \(.*\)$" nil t)
          (string-trim (match-string 1))))))

(defun emacs-scraper--write-output (text base-name)
  "Write TEXT to output file derived from BASE-NAME.
Returns the file path." 
  (let ((path (emacs-scraper--target-path base-name)))
    (with-temp-file path
      (insert text)
      (unless (string-suffix-p "\n" text)
        (insert "\n")))
    path))


(defun emacs-scraper-fetch-url (url &optional base-name)
  "Fetch URL and save content into output dir.
BASE-NAME is optional and will be slugified.
Interactive when called with M-x.
Respects `emacs-scraper-save-images` and `emacs-scraper-save-links`."
  (interactive (list (read-string "URL: ") nil))
  (let ((base (emacs-scraper--slugify (or base-name (emacs-scraper--default-base-name url)))))
    (url-retrieve
     url
     (lambda (_status)
       (unwind-protect
           (let* ((ctype (emacs-scraper--content-type))
                  (content (emacs-scraper--extract-content ctype))
                  (path (emacs-scraper--write-output content base)))
             (when emacs-scraper-open-after-save
               (find-file path))
             (message "Saved: %s" path))
         (kill-buffer (current-buffer)))))))

(defun emacs-scraper--parse-urls (raw)
  "Parse RAW input into a list of URLs." 
  (let* ((lines (split-string raw "\n" t "[ \t]+"))
         (flat (mapcan (lambda (line)
                         (split-string line "[ \t]+" t))
                       lines)))
    flat))

(defun emacs-scraper-fetch-urls (raw-urls &optional base-name)
  "Fetch multiple URLs.
RAW-URLS is a whitespace or newline separated string." 
  (interactive (list (read-string "URLs (space/newline separated): ") nil))
  (dolist (u (emacs-scraper--parse-urls raw-urls))
    (emacs-scraper-fetch-url u base-name)))

(defun emacs-scraper-open ()
  "Open an interactive Emacs scraper UI." 
  (interactive)
  (let ((buf (get-buffer-create emacs-scraper--buffer-name)))
    (with-current-buffer buf
      (kill-all-local-variables)
      (let ((inhibit-read-only t))
        (erase-buffer))
      (remove-overlays)
      (widget-insert "Emacs Scraper (text/images/links)\n\n")
            (widget-insert "Save images: ")
            (let ((img-widget (widget-create 'checkbox :value emacs-scraper-save-images)))
              (widget-insert "  Save links: ")
              (let ((lnk-widget (widget-create 'checkbox :value emacs-scraper-save-links)))
      (widget-insert "URLs (one per line or space-separated):\n")
      (let ((urls-widget (widget-create 'editable-text
                                        :size 80
                                        :format "%v\n\n")))
        (widget-insert "Base name (optional):\n")
        (let ((base-widget (widget-create 'editable-field
                                          :size 40
                                          :format "%v\n\n")))
          (widget-insert "Output directory (relative to project root):\n")
          (let ((dir-widget (widget-create 'editable-field
                                           :size 40
                                           :format "%v\n\n"
                                           :value emacs-scraper-output-dir)))
            (widget-insert "Open after save: ")
            (let ((open-widget (widget-create 'checkbox
                                              :value emacs-scraper-open-after-save)))
              (widget-insert "\n\n")
                (widget-create 'push-button
                       :notify (lambda (_)
                         (setq emacs-scraper-output-dir
                           (widget-value dir-widget))
                         (setq emacs-scraper-open-after-save
                           (widget-value open-widget))
                         (setq emacs-scraper-save-images
                           (widget-value img-widget))
                         (setq emacs-scraper-save-links
                           (widget-value lnk-widget))
                         (let ((urls (widget-value urls-widget))
                           (base (widget-value base-widget)))
                           (emacs-scraper-fetch-urls urls (if (string-empty-p base) nil base))))
                       "Scrape")
                (widget-insert "\n"))))))
      (use-local-map widget-keymap)
      (widget-setup)
      (goto-char (point-min))
      (special-mode))
    (pop-to-buffer buf)))

(defvar emacs-scraper-mode-map
  (let ((map (make-sparse-keymap)))
    (define-key map (kbd "C-c s o") #'emacs-scraper-open)
    (define-key map (kbd "C-c s u") #'emacs-scraper-fetch-url)
    (define-key map (kbd "C-c s m") #'emacs-scraper-fetch-urls)
    map)
  "Keymap for `emacs-scraper-mode'.")

(define-minor-mode emacs-scraper-mode
  "Minor mode for Emacs Scraper keybindings."
  :lighter " Scrape"
  :keymap emacs-scraper-mode-map)

(provide 'emacs-scraper)
;;; emacs-scraper.el ends here
