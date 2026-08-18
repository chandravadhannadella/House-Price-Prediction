const form = document.getElementById('prediction-form');
const predictBtn = document.getElementById('predict-btn');
const resetBtn = document.getElementById('reset-btn');
const messageBox = document.getElementById('message');
const resultCard = document.getElementById('result-card');
const resultValue = document.getElementById('result-value');

const celebrationDurationMs = 5000;
let celebrationTimer = null;

function showMessage(text, type = 'error') {
  messageBox.textContent = text;
  messageBox.className = `message ${type}`;
}

function hideMessage() {
  messageBox.textContent = '';
  messageBox.className = 'message';
}

function clearResult() {
  resultCard.classList.add('hidden');
  resultValue.textContent = '₹ 0';
}

function removeCelebration() {
  const existingCelebration = document.getElementById('celebration-container');
  if (existingCelebration) {
    existingCelebration.remove();
  }

  if (celebrationTimer) {
    clearTimeout(celebrationTimer);
    celebrationTimer = null;
  }
}

function triggerCelebration() {
  removeCelebration();

  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const container = document.createElement('div');
  container.id = 'celebration-container';
  container.className = 'celebration-overlay';
  container.setAttribute('aria-hidden', 'true');

  const colors = ['#ff4d4d', '#ffd93d', '#4ade80', '#60a5fa', '#f472b6', '#a78bfa', '#fb923c', '#2dd4bf'];
  const particleCount = prefersReducedMotion ? 18 : 72;

  for (let index = 0; index < particleCount; index += 1) {
    const particle = document.createElement('span');
    const shapeRoll = Math.random();
    const size = (Math.random() * 12 + 6).toFixed(2);
    const left = `${(Math.random() * 100).toFixed(2)}vw`;
    const duration = (prefersReducedMotion ? 2.4 : (Math.random() * 2 + 2.3)).toFixed(2);
    const delay = (Math.random() * 0.6).toFixed(2);
    const drift = `${((Math.random() - 0.5) * 200).toFixed(0)}px`;
    const rotation = `${(Math.random() * 720 - 360).toFixed(0)}deg`;
    const color = colors[index % colors.length];

    particle.className = 'celebration-piece';

    if (shapeRoll < 0.3) {
      particle.classList.add('celebration-piece--square');
    } else if (shapeRoll < 0.6) {
      particle.classList.add('celebration-piece--streamer');
    } else if (shapeRoll < 0.8) {
      particle.classList.add('celebration-piece--ring');
    } else {
      particle.classList.add('celebration-piece--rectangle');
    }

    particle.style.left = left;
    particle.style.setProperty('--size', `${size}px`);
    particle.style.setProperty('--duration', `${duration}s`);
    particle.style.setProperty('--delay', `${delay}s`);
    particle.style.setProperty('--drift', drift);
    particle.style.setProperty('--rotation', rotation);
    particle.style.setProperty('--color', color);
    particle.style.setProperty('--spin', `${(Math.random() * 540 - 270).toFixed(0)}deg`);

    container.appendChild(particle);
  }

  document.body.appendChild(container);

  celebrationTimer = setTimeout(() => {
    removeCelebration();
  }, celebrationDurationMs);
}

function formatIndianCurrency(value) {
  const safeValue = Number(value);
  if (!Number.isFinite(safeValue)) {
    return '₹ 0';
  }

  return new Intl.NumberFormat('en-IN', {
    style: 'currency',
    currency: 'INR',
    maximumFractionDigits: 0,
  }).format(safeValue);
}

function validateNumericField(fieldName, value) {
  if (value === '' || value === null || value === undefined) {
    throw new Error(`${fieldName} is required.`);
  }

  const numericValue = Number(value);
  if (!Number.isFinite(numericValue) || numericValue <= 0) {
    throw new Error(`${fieldName} must be a valid positive number.`);
  }

  return numericValue;
}

async function handleSubmit(event) {
  event.preventDefault();

  hideMessage();
  clearResult();

  const requiredFields = ['Area', 'BHK', 'Bathroom', 'Parking', 'Per_Sqft'];
  const payload = {};

  try {
    for (const field of requiredFields) {
      const inputValue = document.getElementById(field).value;
      payload[field] = validateNumericField(field, inputValue);
    }

    const optionalFields = ['Furnishing', 'Locality', 'Status', 'Transaction', 'Type'];
    for (const field of optionalFields) {
      const value = document.getElementById(field).value;
      if (value !== '') {
        payload[field] = value;
      }
    }

    predictBtn.disabled = true;
    predictBtn.textContent = 'Predicting house price...';

    const response = await fetch('/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    const data = await response.json();

    if (!response.ok || data.status !== 'success') {
      const errorMessage = data.message || 'Please check your input values and try again.';
      throw new Error(errorMessage);
    }

    const predicted = Number(data.predicted_price);
    if (Number.isFinite(predicted) && data.status === 'success') {
      resultValue.textContent = formatIndianCurrency(predicted);
      resultCard.classList.remove('hidden');
      showMessage('Prediction successful.', 'success');
      triggerCelebration();
    } else {
      throw new Error(data.message || 'Please check your input values and try again.');
    }
  } catch (error) {
    showMessage(error.message || 'Please check your input values and try again.');
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = 'Predict House Price';
  }
}

function handleReset() {
  form.reset();
  hideMessage();
  clearResult();
  removeCelebration();
}

form.addEventListener('submit', handleSubmit);
resetBtn.addEventListener('click', handleReset);
