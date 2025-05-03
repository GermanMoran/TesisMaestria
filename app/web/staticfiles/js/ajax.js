
const listarCultivoGET = async(opcion) => {
    try {
        const response = await fetch(`/api/cultivos/get/${opcion}`);
        //const data = await response.json();
        console.log("ajax js");

        
        //if (data.message === "Success") {
        //    console.log("Cultivos: ", data.cultivo);
            //let html = "";
            //ciudades.forEach(ciudad => {
            //    html += `<option value="${ciudad.id}">${ciudad.nombre}</option>`;
            //});
            //document.getElementById("ciudad").innerHTML = html;
        //}
        
    } catch (error) {
        console.error(error);
    }
};

const listarCultivoPOST = async() => {
    try {
        const response = await fetch(`/api/cultivos/post/listar/`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                nombre: "Cultivo 1",
                descripcion: "Descripcion Cultivo 1",
            }),
        });
        const data = await response.json();
        console.log(data);
    } catch (error) {
        console.error(error);
    }
};

window.addEventListener('DOMContentLoaded', async() => {
    const opcion = "Guardar_Editar";
    //await listarCultivoGET(opcion);
    //await listarCultivoPOST();
});