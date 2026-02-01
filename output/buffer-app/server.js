const http = require("http");
const https = require("https");
const path = require("path");
const fs = require("fs");
const { URL } = require("url");

const PORT = process.env.PORT || 3000;
const PUBLIC_DIR = path.join(__dirname, "public");

const MIME_TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
};

function send(res, status, body, headers = {}) {
  res.writeHead(status, {
    "Content-Type": "text/plain; charset=utf-8",
    ...headers,
  });
  res.end(body);
}

function isHttpUrl(value) {
  try {
    const url = new URL(value);
    return url.protocol === "http:" || url.protocol === "https:";
  } catch {
    return false;
  }
}

function serveStatic(req, res) {
  const reqPath = req.url === "/" ? "/index.html" : req.url;
  const safePath = path.normalize(reqPath).replace(/^\.\.(\/|\\)/, "");
  const filePath = path.join(PUBLIC_DIR, safePath);

  fs.stat(filePath, (err, stat) => {
    if (err || !stat.isFile()) {
      return send(res, 404, "Not Found");
    }

    const ext = path.extname(filePath).toLowerCase();
    const contentType = MIME_TYPES[ext] || "application/octet-stream";

    res.writeHead(200, {
      "Content-Type": contentType,
      "Cache-Control": "no-store",
    });

    fs.createReadStream(filePath).pipe(res);
  });
}

function proxyRequest(req, res) {
  const urlParam = new URL(req.url, `http://${req.headers.host}`).searchParams.get("url");

  if (!urlParam || !isHttpUrl(urlParam)) {
    return send(res, 400, "Invalid or missing url parameter");
  }

  const target = new URL(urlParam);
  const client = target.protocol === "https:" ? https : http;

  const proxyReq = client.get(target, (proxyRes) => {
    res.writeHead(proxyRes.statusCode || 200, {
      "Content-Type": proxyRes.headers["content-type"] || "text/plain; charset=utf-8",
      "Access-Control-Allow-Origin": "*",
      "Cache-Control": "no-store",
    });

    proxyRes.pipe(res);
  });

  proxyReq.on("error", (err) => {
    send(res, 502, `Proxy error: ${err.message}`);
  });
}

const server = http.createServer((req, res) => {
  if (req.url.startsWith("/proxy?")) {
    return proxyRequest(req, res);
  }

  return serveStatic(req, res);
});

server.listen(PORT, () => {
  console.log(`Buffer app running on http://localhost:${PORT}`);
});
