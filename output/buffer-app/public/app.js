const urlInput = document.getElementById("urlInput");
const throttleInput = document.getElementById("throttleInput");
const maxTimeInput = document.getElementById("maxTimeInput");
const stallTimeInput = document.getElementById("stallTimeInput");
const autoScrollInput = document.getElementById("autoScroll");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const downloadBtn = document.getElementById("downloadBtn");
const clearBtn = document.getElementById("clearBtn");

const output = document.getElementById("output");
const stateText = document.getElementById("stateText");
const receivedText = document.getElementById("receivedText");
const displayedText = document.getElementById("displayedText");

let controller = null;
let reader = null;
let bufferQueue = "";
let receivedChars = 0;
let displayedChars = 0;
let ticking = false;
let throttleTimer = null;
let maxTimer = null;
let stallTimer = null;

function setState(text) {
  stateText.textContent = text;
}

function updateCounters() {
  receivedText.textContent = receivedChars.toString();
  displayedText.textContent = displayedChars.toString();
}

function autoScroll() {
  if (!autoScrollInput.checked) return;
  output.parentElement.scrollTop = output.parentElement.scrollHeight;
}

function toggleAutoScroll() {
  autoScrollInput.checked = !autoScrollInput.checked;
  autoScroll();
}

function appendToOutput(text) {
  output.textContent += text;
  displayedChars += text.length;
  updateCounters();
  autoScroll();
}

function clearTimers() {
  if (throttleTimer) {
    clearTimeout(throttleTimer);
    throttleTimer = null;
  }
  if (maxTimer) {
    clearTimeout(maxTimer);
    maxTimer = null;
  }
  if (stallTimer) {
    clearTimeout(stallTimer);
    stallTimer = null;
  }
}

function scheduleMaxTimer() {
  const maxSeconds = Math.max(0, Number(maxTimeInput.value) || 0);
  if (maxSeconds === 0) return;
  maxTimer = setTimeout(() => {
    setState("Timed out");
    stopStream();
  }, maxSeconds * 1000);
}

function resetStallTimer() {
  const stallSeconds = Math.max(1, Number(stallTimeInput.value) || 1);
  if (stallTimer) clearTimeout(stallTimer);
  stallTimer = setTimeout(() => {
    setState("Stalled");
    stopStream();
  }, stallSeconds * 1000);
}

function startThrottleLoop() {
  if (ticking) return;
  ticking = true;

  const tick = () => {
    const rate = Math.max(0, Number(throttleInput.value) || 0);

    if (rate === 0) {
      if (bufferQueue.length > 0) {
        appendToOutput(bufferQueue);
        bufferQueue = "";
      }
    } else {
      const sliceSize = Math.max(1, Math.floor(rate / 10));
      const chunk = bufferQueue.slice(0, sliceSize);
      if (chunk.length > 0) {
        appendToOutput(chunk);
        bufferQueue = bufferQueue.slice(chunk.length);
      }
    }

    if (reader || bufferQueue.length > 0) {
      throttleTimer = setTimeout(tick, 100);
    } else {
      ticking = false;
    }
  };

  tick();
}

async function streamUrl(url) {
  controller = new AbortController();
  setState("Connecting");
  scheduleMaxTimer();
  resetStallTimer();

  try {
    const response = await fetch(`/proxy?url=${encodeURIComponent(url)}`, {
      signal: controller.signal,
    });

    if (!response.ok || !response.body) {
      throw new Error(`Request failed (${response.status})`);
    }

    setState("Streaming");
    const decoder = new TextDecoder();
    reader = response.body.getReader();

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      const textChunk = decoder.decode(value, { stream: true });
      receivedChars += textChunk.length;
      bufferQueue += textChunk;
      updateCounters();
      resetStallTimer();
      startThrottleLoop();
    }

    const remaining = decoder.decode();
    if (remaining) {
      receivedChars += remaining.length;
      bufferQueue += remaining;
      updateCounters();
    }

    setState("Finished");
  } catch (error) {
    if (error.name === "AbortError") {
      setState("Stopped");
    } else {
      setState("Error");
      bufferQueue += `\n[Error] ${error.message}\n`;
      startThrottleLoop();
    }
  } finally {
    reader = null;
    controller = null;
    clearTimers();
    stopBtn.disabled = true;
    startBtn.disabled = false;
    downloadBtn.disabled = output.textContent.length === 0 && bufferQueue.length === 0;
  }
}

function stopStream() {
  if (controller) {
    controller.abort();
  }
  clearTimers();
  ticking = false;
}

function clearOutput() {
  output.textContent = "";
  bufferQueue = "";
  receivedChars = 0;
  displayedChars = 0;
  updateCounters();
  setState("Idle");
  downloadBtn.disabled = true;
}

function downloadBuffer() {
  const content = output.textContent + bufferQueue;
  if (!content) return;
  const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "buffered-text.txt";
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

startBtn.addEventListener("click", () => {
  const url = urlInput.value.trim();
  if (!url) {
    setState("Missing URL");
    return;
  }

  clearOutput();
  setState("Starting");
  startBtn.disabled = true;
  stopBtn.disabled = false;
  downloadBtn.disabled = true;
  streamUrl(url);
});

stopBtn.addEventListener("click", () => {
  stopStream();
});

clearBtn.addEventListener("click", () => {
  stopStream();
  clearOutput();
});

downloadBtn.addEventListener("click", () => {
  downloadBuffer();
});

output.addEventListener("click", () => {
  toggleAutoScroll();
});

document.addEventListener("keydown", (event) => {
  const isShortcut = (event.ctrlKey || event.metaKey) && event.shiftKey && event.code === "KeyS";
  if (isShortcut) {
    event.preventDefault();
    toggleAutoScroll();
  }
});

updateCounters();
