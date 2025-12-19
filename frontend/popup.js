document.addEventListener("DOMContentLoaded", async () => {
  // ===== CONFIG (same as your working version) =====
  const API_KEY = 'API_KEY_HERE';  // Replace with your YouTube Data API v3 key
  const API_URL = 'http://localhost:5000';

  // ===== UI =====
  const pageStatus = document.getElementById("pageStatus");
  const statusDot = document.getElementById("statusDot");
  const statusText = document.getElementById("statusText");
  const videoMeta = document.getElementById("videoMeta");

  const btnAnalyze = document.getElementById("btnAnalyze");
  const btnExportHTML = document.getElementById("btnExportHTML");

  const mTotal = document.getElementById("mTotal");
  const mUnique = document.getElementById("mUnique");
  const mAvgLen = document.getElementById("mAvgLen");
  const mSent = document.getElementById("mSent");

  const chartContainer = document.getElementById("chart-container");
  const trendGraphContainer = document.getElementById("trend-graph-container");
  const wordcloudContainer = document.getElementById("wordcloud-container");

  const cAll = document.getElementById("cAll");
  const cPos = document.getElementById("cPos");
  const cNeu = document.getElementById("cNeu");
  const cNeg = document.getElementById("cNeg");

  const sentimentTabs = document.getElementById("sentimentTabs");
  const commentList = document.getElementById("commentList");
  const commentFooterNote = document.getElementById("commentFooterNote");

  // Tabs
  const tabButtons = document.querySelectorAll(".tab");
  const tabContents = document.querySelectorAll(".tabContent");

  // Theme toggle
  const themeToggle = document.getElementById("themeToggle");

  // Modal
  const imgModal = document.getElementById("imgModal");
  const modalImg = document.getElementById("modalImg");
  const modalTitle = document.getElementById("modalTitle");
  const modalClose = document.getElementById("modalClose");

  // ===== STATE =====
  let currentVideoId = null;
  let currentVideoUrl = null;

  let rawComments = [];   // [{text, timestamp, authorId}]
  let predictions = [];   // [{comment, sentiment, timestamp}]
  let sentimentCounts = { "1": 0, "0": 0, "-1": 0 };
  let activeFilter = "all";

  // store chart images for export report
  let chartImgDataUrl = null;
  let trendImgDataUrl = null;
  let wordcloudImgDataUrl = null;

  // ===== UTIL =====
  function setStatus(type, text) {
    statusDot.classList.remove("ok", "warn");
    if (type === "ok") statusDot.classList.add("ok");
    if (type === "warn") statusDot.classList.add("warn");

    statusText.textContent = text;

    if (type === "ok") pageStatus.textContent = "Ready";
    else if (type === "warn") pageStatus.textContent = "Action needed";
    else pageStatus.textContent = "Working…";
  }

  function escapeHtml(str) {
    return String(str)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function safeFilename(base) {
    return base.replace(/[^a-z0-9_\-]+/gi, "_").slice(0, 90);
  }

  function blobToDataURL(blob) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  }

  function ensureAnalyzed() {
    if (!currentVideoId || predictions.length === 0) {
      setStatus("warn", "Analyze a YouTube video first, then export.");
      return false;
    }
    return true;
  }

  function getFilteredPredictions() {
    if (activeFilter === "all") return predictions;
    return predictions.filter(p => String(p.sentiment) === String(activeFilter));
  }

  function formatSentimentBadge(sent) {
    if (sent === "1") return { label: "Positive", cls: "pos" };
    if (sent === "0") return { label: "Neutral", cls: "neu" };
    return { label: "Negative", cls: "neg" };
  }

  function renderComments() {
    const filtered = getFilteredPredictions();
    commentList.innerHTML = "";

    // clean UI: show up to 60 in popup; export includes more via report
    const toShow = filtered.slice(0, 60);

    if (filtered.length === 0) {
      commentList.innerHTML = `<li class="commentItem"><div class="commentText">No comments found for this filter.</div></li>`;
      commentFooterNote.textContent = "";
      return;
    }

    toShow.forEach(item => {
      const badge = formatSentimentBadge(String(item.sentiment));
      const ts = item.timestamp ? new Date(item.timestamp).toLocaleDateString() : "—";

      const html = `
        <li class="commentItem">
          <div class="commentMeta">
            <span class="badge ${badge.cls}">${badge.label}</span>
            <span style="font-size:11px;color:var(--muted);font-weight:800;">${ts}</span>
          </div>
          <div class="commentText">${escapeHtml(item.comment)}</div>
        </li>
      `;
      commentList.insertAdjacentHTML("beforeend", html);
    });

    commentFooterNote.textContent =
      filtered.length > 60
        ? `Showing 60 of ${filtered.length}. Use Export Report for a shareable summary.`
        : `Showing ${filtered.length} comment(s).`;
  }

  function downloadTextFile(filename, content, mime = "text/plain") {
    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);

    chrome.downloads.download({
      url,
      filename,
      saveAs: true
    }, () => {
      setTimeout(() => URL.revokeObjectURL(url), 5000);
    });
  }

  // // ===== THEME TOGGLE =====
  // themeToggle.addEventListener("click", () => {
  //   document.body.classList.toggle("dark");
  //   themeToggle.textContent = document.body.classList.contains("dark") ? "Light mode" : "Dark mode";
  // });


// Load saved theme
  const savedTheme = localStorage.getItem("creatorinsight-theme");
    if (savedTheme === "dark") {
      document.body.classList.add("dark");
      themeToggle.checked = true;
    }

  // Toggle theme
  themeToggle.addEventListener("change", () => {
    if (themeToggle.checked) {
      document.body.classList.add("dark");
      localStorage.setItem("creatorinsight-theme", "dark");
    } else {
      document.body.classList.remove("dark");
      localStorage.setItem("creatorinsight-theme", "light");
    }
  });


  // ===== TABS =====
  tabButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      tabButtons.forEach(b => b.classList.remove("active"));
      tabContents.forEach(c => c.classList.remove("active"));

      btn.classList.add("active");
      document.getElementById(btn.dataset.tab).classList.add("active");
    });
  });

  // ===== MODAL (expand graphs) =====
  function openModal(title, imgSrc) {
    modalTitle.textContent = title;
    modalImg.src = imgSrc;
    imgModal.classList.add("open");
  }

  modalClose.addEventListener("click", () => imgModal.classList.remove("open"));
  imgModal.addEventListener("click", (e) => {
    if (e.target === imgModal) imgModal.classList.remove("open");
  });

  function wireGraphModal(containerEl, title) {
    containerEl.addEventListener("click", (e) => {
      const img = e.target.closest("img");
      if (!img) return;
      openModal(title, img.src);
    });
  }

  wireGraphModal(chartContainer, "Sentiment Distribution");
  wireGraphModal(trendGraphContainer, "Sentiment Trend");
  wireGraphModal(wordcloudContainer, "Top Themes");

  // ===== PAGE DETECTION =====
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    const url = tabs?.[0]?.url || "";
    currentVideoUrl = url;

    const youtubeRegex = /^https:\/\/(?:www\.)?youtube\.com\/watch\?v=([\w-]{11})/;
    const match = url.match(youtubeRegex);

    if (match && match[1]) {
      currentVideoId = match[1];
      videoMeta.textContent = `Video ID: ${currentVideoId}`;
      setStatus("ok", "Ready to analyze this video.");
    } else {
      currentVideoId = null;
      videoMeta.textContent = "Not a valid YouTube video page. Open a video (watch?v=...) to analyze.";
      setStatus("warn", "Open a YouTube video page first.");
    }
  });

  // ===== COMMENT FILTER TABS =====
  sentimentTabs.addEventListener("click", (e) => {
    const btn = e.target.closest("button");
    if (!btn) return;

    activeFilter = btn.getAttribute("data-filter");
    [...sentimentTabs.querySelectorAll("button")].forEach(b => b.classList.remove("active"));
    btn.classList.add("active");

    renderComments();
  });

  // ===== ANALYZE FLOW =====
  btnAnalyze.addEventListener("click", async () => {
    if (!currentVideoId) {
      setStatus("warn", "Open a YouTube video page first.");
      return;
    }

    // reset state/UI
    rawComments = [];
    predictions = [];
    sentimentCounts = { "1": 0, "0": 0, "-1": 0 };
    activeFilter = "all";
    chartImgDataUrl = trendImgDataUrl = wordcloudImgDataUrl = null;

    chartContainer.innerHTML = "";
    trendGraphContainer.innerHTML = "";
    wordcloudContainer.innerHTML = "";

    mTotal.textContent = "—";
    mUnique.textContent = "—";
    mAvgLen.textContent = "—";
    mSent.textContent = "—";

    cAll.textContent = "0";
    cPos.textContent = "0";
    cNeu.textContent = "0";
    cNeg.textContent = "0";

    commentList.innerHTML = "";
    commentFooterNote.textContent = "";

    // reset filter active
    [...sentimentTabs.querySelectorAll("button")].forEach(b => b.classList.remove("active"));
    sentimentTabs.querySelector('button[data-filter="all"]').classList.add("active");

    setStatus("idle", "Fetching comments…");

    rawComments = await fetchComments(currentVideoId);

    if (!rawComments || rawComments.length === 0) {
      setStatus("warn", "No comments found (or API quota / comments disabled).");
      return;
    }

    setStatus("idle", `Fetched ${rawComments.length} comments. Running sentiment analysis…`);

    predictions = await getSentimentPredictions(rawComments);

    if (!predictions) {
      setStatus("warn", "Sentiment API failed. Check backend is running.");
      return;
    }

    // compute counts + metrics
    let totalSentimentScore = 0;
    const sentimentData = [];

    predictions.forEach((item) => {
      const s = String(item.sentiment);
      if (sentimentCounts[s] !== undefined) sentimentCounts[s]++;

      totalSentimentScore += parseInt(item.sentiment, 10);

      sentimentData.push({
        timestamp: item.timestamp,
        sentiment: parseInt(item.sentiment, 10)
      });
    });

    const totalComments = rawComments.length;
    const uniqueCommenters = new Set(rawComments.map(c => c.authorId)).size;

    const totalWords = rawComments.reduce((sum, c) => {
      const words = c.text.split(/\s+/).filter(w => w.length > 0);
      return sum + words.length;
    }, 0);

    const avgWordLength = (totalWords / totalComments).toFixed(2);
    const avgSentimentScore = (totalSentimentScore / totalComments).toFixed(2);
    const normalizedSentimentScore = (((parseFloat(avgSentimentScore) + 1) / 2) * 10).toFixed(2);

    // update UI
    mTotal.textContent = String(totalComments);
    mUnique.textContent = String(uniqueCommenters);
    mAvgLen.textContent = `${avgWordLength} words`;
    mSent.textContent = `${normalizedSentimentScore}/10`;

    cAll.textContent = String(totalComments);
    cPos.textContent = String(sentimentCounts["1"] || 0);
    cNeu.textContent = String(sentimentCounts["0"] || 0);
    cNeg.textContent = String(sentimentCounts["-1"] || 0);

    setStatus("idle", "Generating visuals…");

    chartImgDataUrl = await fetchAndDisplayChart(sentimentCounts);
    trendImgDataUrl = await fetchAndDisplayTrendGraph(sentimentData);
    wordcloudImgDataUrl = await fetchAndDisplayWordCloud(rawComments.map(c => c.text));

    // render comments for default filter
    renderComments();

    setStatus("ok", "Analysis complete. Explore tabs or export report.");
  });

  // ===== EXPORT HTML ONLY =====
  btnExportHTML.addEventListener("click", () => {
    if (!ensureAnalyzed()) return;

    const payload = buildExportPayload();
    const html = buildHtmlReport(payload);

    const blob = new Blob([html], { type: "text/html" });
    const url = URL.createObjectURL(blob);

    // Open in new browser tab
    chrome.tabs.create({ url });

    // Cleanup later
    setTimeout(() => URL.revokeObjectURL(url), 60_000);
  });


  function buildExportPayload() {
    const totalComments = rawComments.length;
    const uniqueCommenters = new Set(rawComments.map(c => c.authorId)).size;

    const totalWords = rawComments.reduce((sum, c) => {
      const words = c.text.split(/\s+/).filter(w => w.length > 0);
      return sum + words.length;
    }, 0);

    const avgWordLength = (totalWords / totalComments).toFixed(2);

    const totalSentimentScore = predictions.reduce((sum, p) => sum + parseInt(p.sentiment, 10), 0);
    const avgSentimentScore = (totalSentimentScore / totalComments).toFixed(2);
    const normalizedSentimentScore = (((parseFloat(avgSentimentScore) + 1) / 2) * 10).toFixed(2);

    return {
      meta: {
        videoId: currentVideoId,
        url: currentVideoUrl,
        exportedAt: new Date().toISOString(),
        filter: activeFilter
      },
      metrics: {
        totalComments,
        uniqueCommenters,
        avgCommentLengthWords: avgWordLength,
        avgSentimentScoreNormalized10: normalizedSentimentScore
      },
      sentimentCounts: { ...sentimentCounts },
      assets: {
        pieChart: chartImgDataUrl,
        trendGraph: trendImgDataUrl,
        wordcloud: wordcloudImgDataUrl
      },
      comments: getFilteredPredictions()
    };
  }

  function buildHtmlReport(payload) {
    const f = payload.meta.filter;
    const filterLabel = f === "all" ? "All" : (f === "1" ? "Positive" : f === "0" ? "Neutral" : "Negative");

    const pie = payload.assets.pieChart ? `<img src="${payload.assets.pieChart}" style="width:100%;border-radius:14px;border:1px solid rgba(15,23,42,0.10);" />` : "";
    const trend = payload.assets.trendGraph ? `<img src="${payload.assets.trendGraph}" style="width:100%;border-radius:14px;border:1px solid rgba(15,23,42,0.10);" />` : "";
    const wc = payload.assets.wordcloud ? `<img src="${payload.assets.wordcloud}" style="width:100%;border-radius:14px;border:1px solid rgba(15,23,42,0.10);" />` : "";

    const rows = payload.comments.slice(0, 200).map((c) => {
      const s = String(c.sentiment);
      const badge = s === "1" ? "Positive" : s === "0" ? "Neutral" : "Negative";
      return `
        <tr>
          <td style="padding:10px;border-bottom:1px solid rgba(15,23,42,0.08);font-weight:800;">${badge}</td>
          <td style="padding:10px;border-bottom:1px solid rgba(15,23,42,0.08);color:#334155;">${escapeHtml(c.timestamp || "")}</td>
          <td style="padding:10px;border-bottom:1px solid rgba(15,23,42,0.08);">${escapeHtml(c.comment)}</td>
        </tr>
      `;
    }).join("");

    return `
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>CreatorInsight AI Report</title>
  <style>
    body{font-family: ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;background:#F7F8FC;margin:0;color:#0F172A;}
    .wrap{max-width:920px;margin:0 auto;padding:24px;}
    .header{display:flex;justify-content:space-between;gap:14px;align-items:flex-start;margin-bottom:18px;}
    .title{font-size:20px;font-weight:950;}
    .sub{font-size:12px;color:#64748B;margin-top:6px;line-height:1.4;}
    .pill{font-size:12px;padding:6px 10px;border-radius:999px;background:rgba(79,70,229,0.12);border:1px solid rgba(79,70,229,0.20);font-weight:900;color:#4f46e5;}
    .card{background:#fff;border:1px solid rgba(15,23,42,0.10);border-radius:16px;padding:14px;box-shadow:0 10px 30px rgba(15,23,42,0.08);margin-bottom:14px;}
    .grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;}
    .metric{border:1px solid rgba(15,23,42,0.10);border-radius:14px;padding:12px;}
    .label{font-size:11px;color:#64748B;font-weight:900;text-transform:uppercase;letter-spacing:0.3px;}
    .value{font-size:18px;font-weight:950;margin-top:6px;}
    h3{margin:0 0 10px 0;font-size:12px;letter-spacing:0.3px;text-transform:uppercase;}
    table{width:100%;border-collapse:collapse;font-size:13px;}
    .note{font-size:12px;color:#64748B;line-height:1.4;margin-top:8px;}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="header">
      <div>
        <div class="title">CreatorInsight AI — Report</div>
        <div class="sub">
          Video ID: <b>${escapeHtml(payload.meta.videoId)}</b><br/>
          Filter: <b>${filterLabel}</b><br/>
          Exported: ${escapeHtml(payload.meta.exportedAt)}<br/>
          URL: ${escapeHtml(payload.meta.url || "")}
        </div>
      </div>
      <div class="pill">CreatorInsight AI</div>
    </div>

    <div class="card">
      <h3>Metrics</h3>
      <div class="grid">
        <div class="metric"><div class="label">Total Comments</div><div class="value">${payload.metrics.totalComments}</div></div>
        <div class="metric"><div class="label">Unique Commenters</div><div class="value">${payload.metrics.uniqueCommenters}</div></div>
        <div class="metric"><div class="label">Avg Comment Length</div><div class="value">${payload.metrics.avgCommentLengthWords} words</div></div>
        <div class="metric"><div class="label">Avg Sentiment</div><div class="value">${payload.metrics.avgSentimentScoreNormalized10}/10</div></div>
      </div>
      <div class="note">Note: Comment table shows up to 200 comments for readability.</div>
    </div>

    <div class="card">
      <h3>Sentiment Distribution</h3>
      ${pie}
    </div>

    <div class="card">
      <h3>Sentiment Trend</h3>
      ${trend}
    </div>

    <div class="card">
      <h3>Top Themes</h3>
      ${wc}
    </div>

    <div class="card">
      <h3>AI Comment Summarization</h3>
      <div class="note">Coming soon: key themes, audience loves, concerns, and creator suggestions.</div>
    </div>

    <div class="card">
      <h3>Comments (sample)</h3>
      <table>
        <thead>
          <tr>
            <th style="text-align:left;padding:10px;border-bottom:1px solid rgba(15,23,42,0.10);">Sentiment</th>
            <th style="text-align:left;padding:10px;border-bottom:1px solid rgba(15,23,42,0.10);">Timestamp</th>
            <th style="text-align:left;padding:10px;border-bottom:1px solid rgba(15,23,42,0.10);">Comment</th>
          </tr>
        </thead>
        <tbody>
          ${rows}
        </tbody>
      </table>
    </div>
  </div>
</body>
</html>
    `.trim();
  }

  // ===== YouTube + Backend Calls (same as before) =====
  async function fetchComments(videoId) {
    let comments = [];
    let pageToken = "";

    try {
      while (comments.length < 500) {
        const response = await fetch(
          `https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId=${videoId}&maxResults=100&pageToken=${pageToken}&key=${API_KEY}`
        );

        const data = await response.json();

        if (!response.ok) {
          console.error("YouTube API error:", data);
          break;
        }

        if (data.items) {
          data.items.forEach(item => {
            const snippet = item.snippet?.topLevelComment?.snippet;
            if (!snippet) return;

            const commentText = snippet.textOriginal || "";
            const timestamp = snippet.publishedAt || "";
            const authorId = snippet.authorChannelId?.value || "Unknown";

            comments.push({ text: commentText, timestamp, authorId });
          });
        }

        pageToken = data.nextPageToken;
        if (!pageToken) break;
      }
    } catch (error) {
      console.error("Error fetching comments:", error);
    }

    return comments;
  }

  async function getSentimentPredictions(comments) {
    try {
      const response = await fetch(`${API_URL}/predict_with_timestamps`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ comments })
      });

      const result = await response.json();

      if (response.ok) return result;

      console.error("Prediction API error:", result);
      return null;
    } catch (error) {
      console.error("Error fetching predictions:", error);
      return null;
    }
  }

  async function fetchAndDisplayChart(sentimentCounts) {
    try {
      const response = await fetch(`${API_URL}/generate_chart`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sentiment_counts: sentimentCounts })
      });

      if (!response.ok) throw new Error("Failed to fetch chart image");

      const blob = await response.blob();
      const imgURL = URL.createObjectURL(blob);

      const img = document.createElement("img");
      img.src = imgURL;

      chartContainer.innerHTML = "";
      chartContainer.appendChild(img);

      return await blobToDataURL(blob);
    } catch (error) {
      console.error("Error fetching chart image:", error);
      return null;
    }
  }

  async function fetchAndDisplayTrendGraph(sentimentData) {
    try {
      const response = await fetch(`${API_URL}/generate_trend_graph`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sentiment_data: sentimentData })
      });

      if (!response.ok) throw new Error("Failed to fetch trend graph image");

      const blob = await response.blob();
      const imgURL = URL.createObjectURL(blob);

      const img = document.createElement("img");
      img.src = imgURL;

      trendGraphContainer.innerHTML = "";
      trendGraphContainer.appendChild(img);

      return await blobToDataURL(blob);
    } catch (error) {
      console.error("Error fetching trend graph image:", error);
      return null;
    }
  }

  async function fetchAndDisplayWordCloud(comments) {
    try {
      const response = await fetch(`${API_URL}/generate_wordcloud`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ comments })
      });

      if (!response.ok) throw new Error("Failed to fetch word cloud image");

      const blob = await response.blob();
      const imgURL = URL.createObjectURL(blob);

      const img = document.createElement("img");
      img.src = imgURL;

      wordcloudContainer.innerHTML = "";
      wordcloudContainer.appendChild(img);

      return await blobToDataURL(blob);
    } catch (error) {
      console.error("Error fetching word cloud image:", error);
      return null;
    }
  }
});
