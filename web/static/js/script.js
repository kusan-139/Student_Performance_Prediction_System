document.addEventListener('DOMContentLoaded', () => {
    const themeSelect = document.getElementById('theme');
    const body = document.body;

    // Theme Switcher
    themeSelect.addEventListener('change', (e) => {
        body.setAttribute('data-theme', e.target.value);
    });

    // Tab Logic
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // Remove active class from all
            tabBtns.forEach(b => b.classList.remove('active'));
            tabPanes.forEach(p => p.classList.remove('active'));

            // Add active class to clicked tab and its content
            btn.classList.add('active');
            const targetId = btn.getAttribute('data-target');
            document.getElementById(targetId).classList.add('active');
        });
    });

    // Single Prediction Logic
    const singleForm = document.getElementById('single-predict-form');
    const btnRandom = document.getElementById('btn-random-student');
    const singleResult = document.getElementById('single-result');
    const predScore = document.getElementById('pred-score');

    btnRandom.addEventListener('click', async () => {
        try {
            const res = await fetch('/random_student');
            const data = await res.json();
            if (data.error) throw new Error(data.error);

            // Populate form
            for (const key in data) {
                const input = singleForm.elements[key];
                if (input) {
                    input.value = data[key];
                }
            }
        } catch (err) {
            alert('Failed to load random student: ' + err.message);
        }
    });

    singleForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData(singleForm);
        const data = Object.fromEntries(formData.entries());

        // Convert numerics
        data.AttendanceRate = Number(data.AttendanceRate);
        data.StudyHoursPerWeek = Number(data.StudyHoursPerWeek);
        data.PreviousGrade = Number(data.PreviousGrade);
        data.ExtracurricularActivities = Number(data.ExtracurricularActivities);

        try {
            const res = await fetch('/predict_single', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            const result = await res.json();
            
            if (result.error) throw new Error(result.error);

            predScore.textContent = result.prediction;
            singleResult.classList.remove('hidden');
        } catch (err) {
            alert('Prediction failed: ' + err.message);
        }
    });

    // Batch Processing Logic
    const batchForm = document.getElementById('batch-form');
    const batchResult = document.getElementById('batch-result');
    const tbody = document.querySelector('#results-table tbody');
    const btnExport = document.getElementById('btn-export');
    let batchData = [];

    batchForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData(batchForm);

        try {
            const res = await fetch('/predict_batch', {
                method: 'POST',
                body: formData
            });
            const result = await res.json();
            
            if (result.error) throw new Error(result.error);

            batchData = result.data;
            renderTable(batchData);
            batchResult.classList.remove('hidden');
        } catch (err) {
            alert('Batch processing failed: ' + err.message);
        }
    });

    function renderTable(data) {
        tbody.innerHTML = '';
        const theadRow = document.getElementById('table-header-row');
        theadRow.innerHTML = '';

        if (data.length === 0) return;

        // Generate headers dynamically based on CSV columns
        let columns = Object.keys(data[0]);
        
        // Move Predicted_FinalGrade to the end
        if (columns.includes('Predicted_FinalGrade')) {
            columns = columns.filter(c => c !== 'Predicted_FinalGrade');
            columns.push('Predicted_FinalGrade');
        }

        columns.forEach(col => {
            const th = document.createElement('th');
            th.textContent = col;
            theadRow.appendChild(th);
        });

        // Generate rows dynamically
        data.forEach(row => {
            const tr = document.createElement('tr');
            columns.forEach(col => {
                const td = document.createElement('td');
                td.textContent = row[col];
                
                // Highlight the prediction column
                if (col === 'Predicted_FinalGrade') {
                    td.style.fontWeight = 'bold';
                    td.style.color = 'var(--accent)';
                }
                tr.appendChild(td);
            });
            tbody.appendChild(tr);
        });
    }

    // Export logic
    btnExport.addEventListener('click', () => {
        if (!batchData.length) return;
        
        const headers = Object.keys(batchData[0]).join(',');
        const rows = batchData.map(obj => Object.values(obj).join(',')).join('\n');
        const csvContent = `${headers}\n${rows}`;
        
        const blob = new Blob([csvContent], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'batch_predictions.csv';
        a.click();
        URL.revokeObjectURL(url);
    });
});
