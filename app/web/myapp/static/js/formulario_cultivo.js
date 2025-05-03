document.addEventListener('DOMContentLoaded', function() {

    const cultivoInput = document.getElementById('id_input_predict_form');
    const formContainer = document.getElementById('form-container');

    async function loadFormForCultivo(idLote) {
        try {
            const response = await fetch('/api/post/obtener_cultivo/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ id_lote: idLote }),
            });
            const data = await response.json();
            if (data.status === 'success') {
                // Actualiza solo el contenido interno del formulario
                formContainer.innerHTML = data.form_html;
                AccordionSystem.reinitAfterContentLoad(formContainer);
            } else {
                formContainer.innerHTML = `<legend>Error</legend><p>${data.message}</p>`;
            }
        } catch (error) {
            console.error('Error en AJAX:', error);
            formContainer.innerHTML = `<legend>Error</legend><p>Ocurrió un error al cargar los datos.</p>`;
        }
    }
    
    if (cultivoInput) {
        // Al cambiar el valor (o seleccionar del datalist)
        cultivoInput.addEventListener('change', function () {
            const idLote = cultivoInput.value.trim();
            if (idLote) {
                loadFormForCultivo(idLote);
            }
        });
    
        // También al presionar Enter
        cultivoInput.addEventListener('keydown', function (e) {
            if (e.key === 'Enter') {
                e.preventDefault(); // Evita recargar la página
                const idLote = cultivoInput.value.trim();
                if (idLote) {
                    loadFormForCultivo(idLote);
                }
            }
        });
    }
});
