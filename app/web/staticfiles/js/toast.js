const notifications = document.querySelector('.notifications');
const success = document.getElementById('success');
const error = document.getElementById('error');

function createToast(type, icon, title, text) {
    let newToast = document.createElement('div');
    newToast.innerHTML = `
        <div class="toast ${type}">
            <i class="${icon}"></i>
            <div class="content">
                <div class="title">${title}</div>
                <span>${text}</span>
            </div>
            <i class="fa-solid fa-circle-xmark" onclick="(this.parentElement).remove()"></i>
        </div>`;
    notifications.appendChild(newToast);
    newToast.timeOut = setTimeout(
        () => newToast.remove(), 3000
    )
};

document.addEventListener('DOMContentLoaded', function() {
    if (success) {
        success.addEventListener('click', function() {
            console.log('toast click success');
            let type = 'success';
            let icon = 'fa-solid fa-circle-check';
            let title = 'Success';
            let text = 'This is a success toast.';
            createToast(type, icon, title, text);
        });
    }
    if (error) {
        error.addEventListener('click', function() {
            console.log('toast error click');
            let type = 'error';
            let icon = 'fa-solid fa-circle-exclamation';
            let title = 'Error';
            let text = 'This is a error toast.';
            createToast(type, icon, title, text);
    });
    }
});




