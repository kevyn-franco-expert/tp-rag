const API_BASE = '/api/v1';

function showTab(tabName) {
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelectorAll('.tab-button').forEach(button => {
        button.classList.remove('active');
    });
    
    document.getElementById(tabName).classList.add('active');
    event.target.classList.add('active');
}

function showLoading() {
    document.getElementById('loading').classList.add('show');
}

function hideLoading() {
    document.getElementById('loading').classList.remove('show');
}

document.getElementById('search-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const query = document.getElementById('search-query').value.trim();
    if (!query) return;
    
    const resultsDiv = document.getElementById('search-results');
    showLoading();
    
    try {
        const response = await fetch(`${API_BASE}/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        
        const results = await response.json();
        displaySearchResults(results);
        
    } catch (error) {
        resultsDiv.innerHTML = `
            <div class="error">
                <strong>Search Error:</strong> ${error.message}
            </div>
        `;
    } finally {
        hideLoading();
    }
});

document.getElementById('guidance-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const patientContext = document.getElementById('patient-context').value.trim();
    const therapistQuestion = document.getElementById('therapist-question').value.trim();
    
    if (!patientContext || !therapistQuestion) return;
    
    const resultsDiv = document.getElementById('guidance-results');
    showLoading();
    
    try {
        const response = await fetch(`${API_BASE}/guidance`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                patient_context: patientContext,
                therapist_question: therapistQuestion
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        
        const guidance = await response.json();
        displayGuidanceResults(guidance);
        
    } catch (error) {
        resultsDiv.innerHTML = `
            <div class="error">
                <strong>Guidance Error:</strong> ${error.message}
            </div>
        `;
    } finally {
        hideLoading();
    }
});

function displaySearchResults(data) {
    const resultsDiv = document.getElementById('search-results');
    
    const results = data.results || data || [];
    
    if (!results || !Array.isArray(results) || results.length === 0) {
        resultsDiv.innerHTML = `
            <div class="result-card">
                <h3>No Results Found</h3>
                <p>Try different keywords or reduce specificity.</p>
            </div>
        `;
        return;
    }
    
    let html = `<h3>${results.length} Similar Cases Found</h3>`;
    
    results.forEach((result, index) => {
        const score = Math.round(result.similarity * 100);
        html += `
            <div class="result-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <h4>Case ${index + 1}</h4>
                    <span class="similarity-score">${score}% Match</span>
                </div>
                
                <div style="margin-bottom: 15px;">
                    <strong>Context:</strong>
                    <p>${truncateText(result.context, 200)}</p>
                </div>
                
                <div>
                    <strong>Response:</strong>
                    <p>${truncateText(result.response, 200)}</p>
                </div>
            </div>
        `;
    });
    
    resultsDiv.innerHTML = html;
}

function displayGuidanceResults(guidance) {
    const resultsDiv = document.getElementById('guidance-results');
    
    let html = `
        <div class="result-card">
            <h3>Therapeutic Guidance</h3>
            <div style="margin: 15px 0;">
                ${guidance.guidance.replace(/\n/g, '<br>')}
            </div>
        </div>
    `;
    
    if (guidance.similar_cases && guidance.similar_cases.length > 0) {
        html += `
            <div class="result-card">
                <h4>Related Cases (${guidance.similar_cases.length})</h4>
        `;
        
        guidance.similar_cases.forEach((case_item, index) => {
            const score = Math.round(case_item.similarity * 100);
            html += `
                <div style="border-left: 3px solid #3498db; padding-left: 15px; margin: 10px 0;">
                    <div style="font-weight: bold; margin-bottom: 5px;">
                        Case ${index + 1} (${score}% similar)
                    </div>
                    <p style="font-size: 14px; color: #666;">
                        ${truncateText(case_item.context, 150)}
                    </p>
                </div>
            `;
        });
        
        html += `</div>`;
    }
    
    resultsDiv.innerHTML = html;
}

function truncateText(text, maxLength) {
    if (!text) return '';
    return text.length <= maxLength ? text : text.substring(0, maxLength) + '...';
}