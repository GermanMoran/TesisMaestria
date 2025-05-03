# Usar una imagen base de Python 3.10 (ligera)
FROM python:3.10-slim

# Establecer variables de entorno para Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV TZ=America/Bogota

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && \
    apt-get install -y curl gcc graphviz graphviz-dev libpq-dev python3-dev tzdata && \
    ln -fs /usr/share/zoneinfo/America/Bogota /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    rm -rf /var/lib/apt/lists/*

# Copiar el archivo de requisitos primero para evitar reconstruir la imagen innecesariamente
COPY requirements.txt /app/

# Instalar las dependencias de Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar el código Django (el proyecto y demás archivos) al contenedor
COPY . /app/

# Exponer el puerto 8000 (el puerto por defecto de Django)
EXPOSE 8000

# Comando para ejecutar el servidor de desarrollo de Django
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
