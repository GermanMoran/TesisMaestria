
document.addEventListener('DOMContentLoaded', function () {
    const dropdownToggle = document.querySelector('.dropdown-toggle');
    const dropdownMenu = document.querySelector('.dropdown-menu');
    
    const cultivoInput = document.getElementById('id_select_cultivo_optimizar');
    const cultivoFieldset = document.getElementById('cultivo-data-fieldset');
    
    const optimizarBtn = document.getElementById('id-optimizar-btn');
    const precioVentaInput = document.getElementById('id_precio_venta');
    const presupuestoInput = document.getElementById('id_presupuesto');
    
    
    //const costosInputs = document.querySelectorAll('.fieldset-content input[type="number"]');
    //console.log("Cantidad de inputs de costos:", costosInputs.length);
    
    if (cultivoInput && cultivoInput.value.trim() !== '') {
        fetchCultivoData();
        loadCostosUnitarios();
    }

    async function loadCostosUnitarios() {
        try {
            const response = await fetch('/obtener_costos_unitarios/', {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            const data = await response.json();
            if (data.status === 'success') {
                // Actualiza el contenido del contenedor con el HTML recibido
                document.getElementById('costos-container').innerHTML = data.html;
                if (window.AccordionSystem) {
                    AccordionSystem.reinitAfterContentLoad(document.getElementById('costos-container'));
                }

                // Verificar que los inputs se cargaron correctamente
                setTimeout(() => {
                    const inputs = document.querySelectorAll('#costos-container .fieldset-content input[type="number"]');
                    console.log("Cantidad de inputs de costos cargados:", inputs.length);
                }, 100);
                
                return true;
            } else {
                document.getElementById('costos-container').innerHTML = `<p>Error: ${data.message}</p>`;
                return false;
            }
        } catch (error) {
            console.error('Error al cargar Costos Unitarios:', error);
            document.getElementById('costos-container').innerHTML = `<p>Ocurrió un error al cargar los datos.</p>`;
            return false;
        }
    }

    async function fetchCultivoData() {
        const idLote = cultivoInput.value.trim();
        if (!idLote) return;
        try {
            const response = await fetch(`/api/post/obtener_cultivo_variables_manejo/`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ id_lote: idLote }),
            });
            const data = await response.json();
            if (data.status === "success") {
                cultivoFieldset.innerHTML = data.form_html;
                if (window.AccordionSystem) {
                    AccordionSystem.reinitAfterContentLoad(cultivoFieldset);
                }
            } else {
                cultivoFieldset.innerHTML = `<legend>Error</legend><p>${data.message}</p>`;
            }
        } catch (error) {
            console.error("Error en AJAX:", error);
            cultivoFieldset.innerHTML = `<legend>Error</legend><p>Ocurrió un error al cargar los datos.</p>`;
        }
    }

    if (cultivoInput) {
        cultivoInput.addEventListener('keydown', function (e) {
            if (e.key === "Enter") {
                e.preventDefault();
                fetchCultivoData();
            }
        });
        cultivoInput.addEventListener('change', fetchCultivoData);
    }

    if (dropdownToggle && dropdownMenu) {
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
    }


    if (optimizarBtn) {
        optimizarBtn.addEventListener('click', async function (e) {
            e.preventDefault();
    
            const selectedCultivo = cultivoInput.value.trim();
            const precioVenta = precioVentaInput.value;
            const presupuesto = presupuestoInput.value;

            const costosInputs = document.querySelectorAll('#costos-container .fieldset-content input[type="number"]');
            console.log("Cantidad de inputs de costos al enviar:", costosInputs.length);
            
            const costosUnitarios = {};
            costosInputs.forEach(input => {
                costosUnitarios[input.name] = Number(input.value);
            });
            
            console.log("Costos unitarios:", costosUnitarios);

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
                    //cultivoFieldset.insertAdjacentHTML('afterbegin', data.message)
                    cultivoFieldset.innerHTML = data.message;
                    console.log("click optimizar success");
                    if (window.AccordionSystem) {
                        AccordionSystem.reinitAfterContentLoad(cultivoFieldset);
                    }
                } else {
                    cultivoFieldset.innerHTML = `<legend>Error</legend><p>${data.message}</p>`;
                }
            } catch (error) {
                console.error("Error al optimizar:", error);
                cultivoFieldset.innerHTML = `<legend>Error</legend><p>Ocurrió un error al optimizar.</p>`;
            }
        });
    }
});
