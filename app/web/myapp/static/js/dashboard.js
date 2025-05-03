document.addEventListener('DOMContentLoaded', function () {
    // Handle dropdown change without page reload
    const loteDropdown = document.getElementById('lote-dropdown');

    if (loteDropdown) {
        loteDropdown.addEventListener('change', function (e) {
            e.preventDefault();
            const selectedLote = this.value;
            updateDashboard(selectedLote);
            console.log('Selected Lote:', selectedLote);
        });
    }

    function updateDashboard(loteId) {
        // Show loading indicators
        document.querySelectorAll('.graph-container').forEach(container => {
            container.innerHTML = '<div class="text-center p-5"><div class="spinner-border" role="status"><span class="visually-hidden">Loading...</span></div></div>';
        });

        // Make AJAX request
        fetch(`/api/dashboard-data/?lote_id=${loteId}`)
            .then(response => response.json())
            .then(data => {
                console.log('Dashboard data yield chart:', data.soil_chart);

                document.getElementById('total-nitrogeno').textContent = data.total_nitrogeno + ' (Kg)';
                document.getElementById('total-fosforo').textContent = data.total_fosforo + ' (Kg)';
                document.getElementById('total-potasio').textContent = data.total_potasio + ' (Kg)';
                document.getElementById('total-rendimiento').textContent = data.rendimiento_prom + ' (Kg/ha)';

                document.getElementById('yield-chart-container').innerHTML = data.yield_chart;
                document.getElementById('soil-chart-container').innerHTML = data.soil_chart;
                document.getElementById('controls-chart-container').innerHTML = data.controls_chart;

                var script = document.createElement('script');
                script.type = 'text/javascript';
                script.innerHTML = data.yield_chart.match(/<script.*?>([\s\S]*?)<\/script>/)[1];
                document.body.appendChild(script);

                var soilScript = document.createElement('script');
                soilScript.type = 'text/javascript';
                soilScript.innerHTML = data.soil_chart.match(/<script.*?>([\s\S]*?)<\/script>/)[1];
                document.body.appendChild(soilScript);

                var controlsScript = document.createElement('script');
                controlsScript.type = 'text/javascript';
                controlsScript.innerHTML = data.controls_chart.match(/<script.*?>([\s\S]*?)<\/script>/)[1]; // Extrae el código JS de 'controls_chart'
                document.body.appendChild(controlsScript);

                
                updateTable(data.table_data, data.table_columns);
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Hubo un error al cargar los datos del dashboard');
            });
    }

    function updateTable(data, columns) {
        const tableBody = document.querySelector('#data-table tbody');
        tableBody.innerHTML = '';

        data.forEach(row => {
            const tr = document.createElement('tr');
            columns.forEach(column => {
                const td = document.createElement('td');
                td.textContent = row[column];
                tr.appendChild(td);
            });
            tableBody.appendChild(tr);
        });
    }
});