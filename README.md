# TesisMaestria
## Acerca de este proyecto
Repositorio asociado a todos los archivos obtenidos de la tesis de maestría "Predicción del rendimiento del cultivo del maíz mediante el uso de CLR y metaheurísticas."

## Estructura del Proyecto
```
├── app                <- Codigo fuente del proyecto
├── README.md          <- El archivo README de nivel superior para desarrolladores que 
|                         utilizan este    proyecto.
├── data
│   ├── external       <- Fuentes de datos externas
│   ├── Silver         <- Datos intermedios que han sido procesados y transformados.
│   ├── Gold           <- Dataset Final utilizado  para modelos de ML.
│   └── Bronze         <- Fuentes originales suministradas por la entidad.
│
├── functional_tests   <- Tests Funcionales, Incluye manual de usario con pruebas.
│
├── models             <- Modelos Entrenados (Archivos.pkl) para ser consumidos.
│
├── notebooks          <- Jupyter notebooks para cada fase del proyecto.
│
├── Referencias        <- Diccionarios de datos, manuales y todo otro material explicativo.
│
├── Documentos         <- Generados en la investigación: Monografia y Articulos.
│   └── figuras        <- Generadas en la investigación Monografia y Articulos.
|                       
│
├── Instrucciones.txt  <- Instrucciones a tener en cuenta para el despliegue de
|                          la aplicación WEB.
│
└── 
```
## Herramientas 🛠️
- Python
- Jupyter Notebooks
- Git Hub
- Docker
- Postgres SQL
- Dash Ploty
- Django
- (HTML, Css y JavaScript)

## Pasos para hacer uso del proyecto
1. Clonar el código del repositorio de git
2. Instala WSL para Linux
3. Abrir la linea de comandos de WSL y configurar (user y password) para la primera vez.
4. Instar Docker en Linux siguiendo los siguientes comandos.

    ```
    # Add Docker's official GPG key:
    sudo apt-get update
    sudo apt-get install ca-certificates curl
    sudo install -m 0755 -d /etc/apt/keyrings
    sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    sudo chmod a+r /etc/apt/keyrings/docker.asc

    # Add the repository to Apt sources:
    echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update
    ```
5. Instala paquetes y dependencias de docker 

    ```
    sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    ```

6. Copiar la carpeta app del proyecto a la ruta del WSL

7. Se crea un nuevo grupo en el sistema

    ```
    sudo groupadd docker
    ```

8. (Opcional) Evitar escribir sudo cada vez que ejecuta el comando docker.

    ```
    sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

     ```


## Pasos para ejecutar la app Django localmente

1. Verificar que la  aplicación app este sobre la carpeta ./WSL

2. Ingresamos a la carpeta del proyecto app

    ```
    cd app
    ```



3. Ejecutamos el siguiente comando para lanzar la aplicación Localmente
    ```
    docker-compose up -d 
    ```

4. Visualización de la apliación WEB en el navegador

    ```
    localhost:8000
    ```


5. Para detener la aplicación ejecutamos el siguiente comando 

    ```
    docker-compose down
    ```

6. Verificacion de logs y y eventos del aplicativo
    ```
    docker logs -f django-web
    ```
