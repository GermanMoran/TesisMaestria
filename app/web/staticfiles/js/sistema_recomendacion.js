
document.addEventListener('DOMContentLoaded', function () {
    const dropdownToggle = document.querySelector('.dropdown-toggle');
    const dropdownMenu = document.querySelector('.dropdown-menu');

    const cultivoInput = document.getElementById('id_select_cultivo_optimizar');
    const cultivoFieldset = document.getElementById('cultivo-data-fieldset');

    const optimizarBtn = document.getElementById('id-optimizar-btn');
    const precioVentaInput = document.getElementById('id_precio_venta');
    const presupuestoInput = document.getElementById('id_presupuesto');

    
    const costosInputs = document.querySelectorAll('.dropdown-menu input[type="number"]');
    console.log("Cantidad de inputs de costos:", costosInputs.length);

    if (cultivoInput.value.trim() !== '') {
        fetchCultivoData();
    }

    async function fetchCultivoData() {
        const idLote = cultivoInput.value.trim();
        if (!idLote) return;
        try {
            const response = await fetch(`/api/post/obtener_cultivo/`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ id_lote: idLote }),
            });
            const data = await response.json();
            if (data.status === "success") {
                cultivoFieldset.innerHTML = data.form_html;
            } else {
                cultivoFieldset.innerHTML = `<legend>Error</legend><p>${data.message}</p>`;
            }
        } catch (error) {
            console.error("Error en AJAX:", error);
            cultivoFieldset.innerHTML = `<legend>Error</legend><p>Ocurrió un error al cargar los datos.</p>`;
        }
    }

    // Ejecuta la función al presionar Enter o cuando cambie el valor del input
    cultivoInput.addEventListener('keydown', function (e) {
        if (e.key === "Enter") {
            e.preventDefault();
            fetchCultivoData();
        }
    });

    cultivoInput.addEventListener('change', fetchCultivoData);

    dropdownToggle.addEventListener('click', function (e) {
        e.stopPropagation();
        dropdownMenu.classList.toggle('show');
    });

    dropdownMenu.addEventListener('click', function (e) {
        e.stopPropagation();
    });

    document.addEventListener('click', function () {
        if (dropdownMenu.classList.contains('show')) {
            dropdownMenu.classList.remove('show');
        }
    });

    optimizarBtn.addEventListener('click', async function (e) {
        e.preventDefault();

        const selectedCultivo = cultivoInput.value.trim();
        const precioVenta = precioVentaInput.value;
        const presupuesto = presupuestoInput.value;
        
        const costosUnitarios = {};
        costosInputs.forEach(input => {
            costosUnitarios[input.name] = input.value;
        });

        const dataToSend = {
            id_lote: selectedCultivo,
            precio_venta: precioVenta,
            presupuesto: presupuesto,
            costos_unitarios: costosUnitarios,
        };

        try {
            const response = await fetch(`/api/post/optimizar/`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(dataToSend),
            });
            const data = await response.json();
            if (data.status === "success") {
                cultivoFieldset.insertAdjacentHTML('afterbegin', data.message)
                console.log("click optimizar success");
            } else {
                cultivoFieldset.innerHTML = `<legend>Error</legend><p>${data.message}</p>`;
            }
        } catch (error) {
            console.error("Error al optimizar:", error);
            cultivoFieldset.innerHTML = `<legend>Error</legend><p>Ocurrió un error al optimizar.</p>`;
        }
    });
});
