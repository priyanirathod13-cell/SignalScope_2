/**
 * SignalScope Frontend Application Controller
 * Connects directly to the live FastAPI backend for authentic inference,
 * Grad-CAM attribution rendering, and interactive explainability.
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('fileInput');
  const browseBtn = document.getElementById('browseBtn');
  const examplesGrid = document.getElementById('examplesGrid');

  const uploadCard = document.getElementById('uploadCard');
  const previewCard = document.getElementById('previewCard');
  const loadingCard = document.getElementById('loadingCard');
  const resultDashboard = document.getElementById('resultDashboard');

  const previewImg = document.getElementById('previewImg');
  const fileNameEl = document.getElementById('fileName');
  const fileDimensionsEl = document.getElementById('fileDimensions');
  const fileSizeEl = document.getElementById('fileSize');
  const removeBtn = document.getElementById('removeBtn');
  const analyzeBtn = document.getElementById('analyzeBtn');
  const newAnalysisBtn = document.getElementById('newAnalysisBtn');

  // Loading Steps
  const step1 = document.getElementById('step1');
  const step2 = document.getElementById('step2');
  const step3 = document.getElementById('step3');
  const step4 = document.getElementById('step4');

  // Verdict elements
  const verdictCard = document.getElementById('verdictCard');
  const verdictTitle = document.getElementById('verdictTitle');
  const verdictSummary = document.getElementById('verdictSummary');
  const verdictScore = document.getElementById('verdictScore');
  const verdictSymbol = document.getElementById('verdictSymbol');
  const verdictPillText = document.getElementById('verdictPillText');
  const aiProbText = document.getElementById('aiProbText');
  const realProbText = document.getElementById('realProbText');
  const aiProbBar = document.getElementById('aiProbBar');
  const realProbBar = document.getElementById('realProbBar');

  // Technical chips
  const techModel = document.getElementById('techModel');
  const techLatency = document.getElementById('techLatency');
  const techDimensions = document.getElementById('techDimensions');
  const techLayer = document.getElementById('techLayer');

  // Image Stage Elements
  const baseImage = document.getElementById('baseImage');
  const overlayImage = document.getElementById('overlayImage');
  const splitOriginal = document.getElementById('splitOriginal');
  const splitHeatmap = document.getElementById('splitHeatmap');
  const splitOverlay = document.getElementById('splitOverlay');
  const singleViewContainer = document.getElementById('singleViewContainer');
  const splitViewContainer = document.getElementById('splitViewContainer');
  const sliderToolbar = document.getElementById('sliderToolbar');
  const opacitySlider = document.getElementById('opacitySlider');
  const opacityValue = document.getElementById('opacityValue');
  const responsibleText = document.getElementById('responsibleText');

  // Error Toast
  const errorToast = document.getElementById('errorToast');
  const errorMessage = document.getElementById('errorMessage');
  const errorCloseBtn = document.getElementById('errorCloseBtn');

  // State
  let currentFile = null;
  let currentResult = null;
  let currentViewMode = 'overlay';

  // Dynamic API host determination (supports integrated :8000 and standalone :3000)
  const API_BASE = (window.location.port === '8000') ? '' : 'http://127.0.0.1:8000';

  // Format Helper
  function formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  }

  function showError(msg) {
    errorMessage.textContent = msg;
    errorToast.style.display = 'flex';
    setTimeout(() => {
      errorToast.style.display = 'none';
    }, 6000);
  }

  errorCloseBtn.addEventListener('click', () => {
    errorToast.style.display = 'none';
  });

  // --- 1. Load Example Gallery from API ---
  async function loadExamples() {
    try {
      const res = await fetch(`${API_BASE}/api/examples`);
      if (!res.ok) return;
      const examples = await res.json();
      examplesGrid.innerHTML = '';

      examples.forEach(ex => {
        const card = document.createElement('div');
        card.className = 'example-card';
        const thumbSrc = ex.thumbnail_url.startsWith('http') ? ex.thumbnail_url : `${API_BASE}${ex.thumbnail_url}`;
        card.innerHTML = `
          <img src="${thumbSrc}" alt="${ex.title}" class="example-thumb">
          <div class="example-meta">
            <span class="example-title">${ex.title}</span>
            <span class="example-source">${ex.source}</span>
          </div>
        `;
        card.addEventListener('click', () => {
          analyzeExample(ex.id, ex.title, thumbSrc);
        });
        examplesGrid.appendChild(card);
      });
    } catch (e) {
      console.warn('Could not fetch examples:', e);
    }
  }

  // --- 2. File Selection & Drag-and-Drop ---
  browseBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  dropZone.addEventListener('click', () => {
    fileInput.click();
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelected(e.target.files[0]);
    }
  });

  ['dragenter', 'dragover'].forEach(name => {
    dropZone.addEventListener(name, (e) => {
      e.preventDefault();
      dropZone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(name => {
    dropZone.addEventListener(name, (e) => {
      e.preventDefault();
      dropZone.classList.remove('dragover');
    });
  });

  dropZone.addEventListener('drop', (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  function handleFileSelected(file) {
    const validExts = ['.jpg', '.jpeg', '.png', '.webp'];
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    if (!validExts.includes(ext)) {
      showError(`Unsupported file format '${ext}'. Allowed: JPG, PNG, WEBP.`);
      return;
    }

    if (file.size > 15 * 1024 * 1024) {
      showError('File size exceeds the 15 MB limit.');
      return;
    }

    currentFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      fileNameEl.textContent = file.name;
      fileSizeEl.textContent = formatBytes(file.size);

      // Read natural dimensions
      const tempImg = new Image();
      tempImg.onload = () => {
        fileDimensionsEl.textContent = `${tempImg.naturalWidth} x ${tempImg.naturalHeight} px`;
      };
      tempImg.src = e.target.result;

      uploadCard.style.display = 'none';
      resultDashboard.style.display = 'none';
      previewCard.style.display = 'block';
    };
    reader.readAsDataURL(file);
  }

  removeBtn.addEventListener('click', resetView);
  newAnalysisBtn.addEventListener('click', resetView);

  function resetView() {
    currentFile = null;
    currentResult = null;
    fileInput.value = '';
    previewImg.src = '';
    previewCard.style.display = 'none';
    loadingCard.style.display = 'none';
    resultDashboard.style.display = 'none';
    uploadCard.style.display = 'block';
  }

  // --- 3. Run Analysis ---
  analyzeBtn.addEventListener('click', () => {
    if (!currentFile) return;
    executeAnalysis(currentFile);
  });

  async function analyzeExample(exampleId, title, thumbUrl) {
    uploadCard.style.display = 'none';
    previewCard.style.display = 'none';
    resultDashboard.style.display = 'none';
    loadingCard.style.display = 'block';
    runScanningAnimation();

    try {
      const res = await fetch(`${API_BASE}/api/predict-example`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ example_id: exampleId })
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || 'Analysis request failed.');
      }

      const data = await res.json();
      renderResults(data);
    } catch (err) {
      loadingCard.style.display = 'none';
      uploadCard.style.display = 'block';
      showError(err.message || 'Error executing example analysis.');
    }
  }

  async function executeAnalysis(file) {
    previewCard.style.display = 'none';
    resultDashboard.style.display = 'none';
    loadingCard.style.display = 'block';
    runScanningAnimation();

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${API_BASE}/api/predict`, {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || 'Analysis request failed.');
      }

      const data = await res.json();
      renderResults(data);
    } catch (err) {
      loadingCard.style.display = 'none';
      uploadCard.style.display = 'block';
      showError(err.message || 'Inference failed. Please verify the image file.');
    }
  }

  function runScanningAnimation() {
    const steps = [step1, step2, step3, step4];
    steps.forEach(s => { s.className = 'scan-step'; });
    
    steps[0].className = 'scan-step active';
    setTimeout(() => {
      steps[0].className = 'scan-step done';
      steps[1].className = 'scan-step active';
    }, 180);
    setTimeout(() => {
      steps[1].className = 'scan-step done';
      steps[2].className = 'scan-step active';
    }, 360);
    setTimeout(() => {
      steps[2].className = 'scan-step done';
      steps[3].className = 'scan-step active';
    }, 540);
  }

  // --- 4. Render Analysis & Explainability Results ---
  function renderResults(data) {
    currentResult = data;
    loadingCard.style.display = 'none';
    resultDashboard.style.display = 'block';

    const isReal = data.label === 'REAL';
    
    // Verdict Card styling
    verdictCard.className = `glass-card verdict-card ${isReal ? 'verdict-real' : 'verdict-ai'}`;
    verdictSymbol.textContent = isReal ? '✓' : '⚠';
    verdictTitle.textContent = isReal ? 'REAL PHOTOGRAPHY' : 'AI-GENERATED';
    verdictSummary.textContent = isReal
      ? 'Visual artifacts show high natural sensor continuity with absence of diffusion grid frequencies.'
      : 'Visual feature gradients exhibit strong alignment with generative diffusion artifacts.';

    verdictScore.textContent = data.confidence_percent;
    verdictPillText.textContent = isReal ? 'VERIFIED REAL' : 'SYNTHETIC DETECTED';

    // Probabilities
    const aiPct = (data.probabilities.synthetic * 100).toFixed(1);
    const realPct = (data.probabilities.real * 100).toFixed(1);

    aiProbText.textContent = `${aiPct}%`;
    realProbText.textContent = `${realPct}%`;
    aiProbBar.style.width = `${aiPct}%`;
    realProbBar.style.width = `${realPct}%`;

    // Technical chips
    techModel.textContent = data.model;
    techLatency.textContent = `${Math.round(data.inference_time * 1000)} ms`;
    techDimensions.textContent = `${data.image_width} x ${data.image_height} px`;
    techLayer.textContent = data.target_layer || 'backbone.features[8]';

    // Populate Images
    baseImage.src = data.original_image;
    overlayImage.src = data.overlay_image;

    splitOriginal.src = data.original_image;
    splitHeatmap.src = data.heatmap_image;
    splitOverlay.src = data.overlay_image;

    responsibleText.textContent = data.responsible_explanation;

    // Reset view mode to overlay
    setViewMode('overlay');
    opacitySlider.value = 50;
    opacityValue.textContent = '50%';
    overlayImage.style.opacity = 0.5;

    // Scroll smoothly to result
    resultDashboard.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  // --- 5. Interactive View Modes & Transparency Slider ---
  const viewTabs = document.querySelectorAll('.view-tab');
  viewTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      viewTabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      setViewMode(tab.dataset.mode);
    });
  });

  function setViewMode(mode) {
    currentViewMode = mode;
    if (!currentResult) return;

    if (mode === 'overlay') {
      singleViewContainer.style.display = 'flex';
      splitViewContainer.style.display = 'none';
      sliderToolbar.style.display = 'flex';
      baseImage.src = currentResult.original_image;
      overlayImage.style.display = 'block';
      overlayImage.src = currentResult.overlay_image;
    } else if (mode === 'heatmap') {
      singleViewContainer.style.display = 'flex';
      splitViewContainer.style.display = 'none';
      sliderToolbar.style.display = 'none';
      baseImage.src = currentResult.heatmap_image;
      overlayImage.style.display = 'none';
    } else if (mode === 'original') {
      singleViewContainer.style.display = 'flex';
      splitViewContainer.style.display = 'none';
      sliderToolbar.style.display = 'none';
      baseImage.src = currentResult.original_image;
      overlayImage.style.display = 'none';
    } else if (mode === 'split') {
      singleViewContainer.style.display = 'none';
      splitViewContainer.style.display = 'grid';
      sliderToolbar.style.display = 'none';
    }
  }

  opacitySlider.addEventListener('input', (e) => {
    const val = e.target.value;
    opacityValue.textContent = `${val}%`;
    overlayImage.style.opacity = val / 100.0;
  });

  // --- 6. Back to Top Smooth Navigation ---
  const backToTopBtn = document.getElementById('backToTopBtn');
  if (backToTopBtn) {
    backToTopBtn.addEventListener('click', (e) => {
      e.preventDefault();
      window.scrollTo({
        top: 0,
        behavior: 'smooth'
      });
    });
  }

  // Initial load
  loadExamples();
});
