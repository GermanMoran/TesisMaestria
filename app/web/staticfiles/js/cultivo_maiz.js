const modal = document.getElementById('modal');
const cultivoTable = document.getElementById('cultivo-maiz-table');
const hideModalBtn = document.getElementById('close-modal');
const agregarBtnTable = document.getElementById('agregar-btn-table');

const cancelarBtnFormulario = async () => {
    hideModal();
}

const guardarBtnFormulario = async () => {
    console.log('Guardar Btn Formulario Cultivo');

    const sourceInput = document.querySelector('input[name="source"]');
    const source = sourceInput.value;
    console.log(source);

    const [accion, pk] = source.split('-');

    const form = document.querySelector('#form-modal-container form');
    if (!form) {
        console.error("No se encontró el formulario");
        return;
    }

    const formData = new FormData(form);
    const dataToSend = {};
    formData.forEach((value, key) => {
        dataToSend[key] = value;
    });

    console.log("Accion: ", accion);
    console.log("pk: ", pk);
    console.log("datos del formulario: ", dataToSend);

    if (pk) {
        try {
            const response = await fetch(`/api/post/editar_formulario_cultivo/${pk}/`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(dataToSend),
            });

            const data = await response.json();
            console.log("Respuesta de edición:", data);

            if (data.status === "success") {
                localStorage.setItem("toastNotification", JSON.stringify({
                    type: "success",              // o el tipo que necesites
                    icon: "fa-solid fa-check",    // el ícono deseado
                    title: "Editar",             // título de la notificación
                    text: "Cultivo actualizado correctamente" // mensaje
                }));
                window.location.href = "/cultivo_maiz_tabla/";
            } else {
                console.error("Error al editar el formulario: ", data);
            }
        } catch (error) {
            console.error("Error al editar el formulario: ", error);
        }
    } else {
        try {
            const response = await fetch(`/api/post/editar_formulario_cultivo/`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(dataToSend),
            });

            const data = await response.json();
            console.log("Respuesta de edición:", data);

            if (data.status === "success") {
                localStorage.setItem("toastNotification", JSON.stringify({
                    type: "success",              // o el tipo que necesites
                    icon: "fa-solid fa-check",    // el ícono deseado
                    title: "Guardar",             // título de la notificación
                    text: "Cultivo agregado correctamente" // mensaje
                }));
                window.location.href = "/cultivo_maiz_tabla/";
            } else {
                console.error("Error al crear el formulario: ", data);
            }
        } catch (error) {
            console.error("Error al editar el formulario: ", error);
        }
    }
};

async function eliminarCultivo(pk) {
    console.log('Eliminar Cultivo', pk);

    if (!confirm("¿Está seguro de eliminar este cultivo?")) {
        return;
    }

    try {
        const response = await fetch(`/api/post/eliminar_formulario_cultivo/${pk}/`, {
            method: "DELETE",  
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                nombre: "eliminar cultivo",
            }),
        });

        const data = await response.json();
        console.log("Respuesta de eliminación:", data);

        if (data.status === "success") {
            // Guardar la notificación en localStorage para que se muestre en la siguiente página
            localStorage.setItem("toastNotification", JSON.stringify({
                type: "success",              // o el tipo que necesites
                icon: "fa-solid fa-check",    // el ícono deseado
                title: "Eliminar",            // título de la notificación
                text: "Cultivo eliminado correctamente" // mensaje
            }));
            window.location.href = "/cultivo_maiz_tabla/";
        } else {
            console.error("Error al eliminar el cultivo:", data);
        }
    } catch (error) {
        console.error("Error al eliminar el cultivo:", error);
    }
}


const editarCultivo = async (pk) => {
    console.log('Editar Cultivo', pk);
    agregarBtnTable.style.display = 'none';

    try {
        const response = await fetch(`/api/post/formulario_cultivo/${pk}/`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                nombre: "editar cultivo",
            }),
        });

        const data = await response.json();
        console.log(data);

        const hiddenInput = document.createElement('input');
        hiddenInput.setAttribute('type', 'hidden');
        hiddenInput.setAttribute('name', 'source');
        hiddenInput.setAttribute('value', `editarCultivo-${pk}`);

        const modalForm = document.getElementById('form-modal-container');
        modalForm.innerHTML = data.form_html;
        modalForm.appendChild(hiddenInput);

    } catch (error) {
        console.error(error);
    }

    showModal();
}

const agregarCultivo = async () => {
    console.log('Agregar Cultivo');
    agregarBtnTable.style.display = 'none';

    try {
        const response = await fetch(`/api/post/formulario_cultivo/`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                nombre: "agregar cultivo",
            }),
        });

        const data = await response.json();
        console.log(data);
        const hiddenInput = document.createElement('input');
        hiddenInput.setAttribute('type', 'hidden');
        hiddenInput.setAttribute('name', 'source');
        hiddenInput.setAttribute('value', 'agregarCultivo');

        const modalForm = document.getElementById('form-modal-container');
        modalForm.innerHTML = data.form_html;
        modalForm.appendChild(hiddenInput);

    } catch (error) {
        console.error(error);
    }

    showModal();
}

const showModal = () => {
    modal.style.display = 'block';
    cultivoTable.style.display = 'none';
}

const hideModal = () => {
    modal.style.display = 'none';
    cultivoTable.style.display = 'block';
    agregarBtnTable.style.display = 'block';
}

const listarCultivoTablaPOST = async () => {

    try {
        const response = await fetch(`/api/post/cultivos/listar/`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                nombre: "cultivo-maiz-table",
            }),
        });

        const data = await response.json();

        console.log(data);
        cultivos_data_json = JSON.parse(data.cultivos_data);
        console.log(cultivos_data_json[0].fields);

        if (cultivos_data_json.length > 0) {
            // Obtenemos la lista de cabeceras a partir del primer registro
            const headerRow = cultivoTable.querySelector('thead');
            const bodyRow = cultivoTable.querySelector('tbody');
            const cabeceras = Object.keys(cultivos_data_json[0].fields);

            headerRow.innerHTML = `
            <tr>
            <th colspan="2">Opciones</th>
            ${cabeceras.map(cabecera => `<th>${cabecera}</th>`).join('')}
            </tr>
            `;
            let filasHTML = "";
            cultivos_data_json.forEach(item => {
                // Obtenemos los valores en el mismo orden que las cabeceras
                filasHTML += `
                <tr>
                <td><button type="button" class="btn-edit" onclick="editarCultivo(${item.pk})">Editar</button></td>
                <td><button type="button" class="btn-delete" onclick="eliminarCultivo(${item.pk})">Eliminar</button></td>
                ${cabeceras.map(key => `<td>${item.fields[key]}</td>`).join('')}
                </tr>
                `;
            });
            bodyRow.innerHTML = filasHTML;

            cultivoTable?.appendChild(headerRow);
            cultivoTable?.appendChild(bodyRow);

        }
    } catch (error) {
        console.error(error);
    }
};



document.addEventListener('DOMContentLoaded', async () => {
    const toastData = localStorage.getItem("toastNotification");
    if (toastData) {
        const { type, icon, title, text } = JSON.parse(toastData);
        createToast(type, icon, title, text);
        localStorage.removeItem("toastNotification");
    }

    if (cultivoTable) {
        listarCultivoTablaPOST();
    }

    agregarBtnTable?.addEventListener('click', agregarCultivo);

    hideModalBtn?.addEventListener('click', hideModal);

});