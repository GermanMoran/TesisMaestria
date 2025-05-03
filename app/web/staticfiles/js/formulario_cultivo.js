
const guardar_btn_form = document.getElementById('guardar-btn-form');

document.addEventListener('DOMContentLoaded', function() {
    if (guardar_btn_form) {
        guardar_btn_form.addEventListener('click', function() {
            console.log('Hola Mundo formulario'); 
        });
    }
});