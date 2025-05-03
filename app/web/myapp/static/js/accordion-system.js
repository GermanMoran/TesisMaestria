/**
 * accordion-system.js - Sistema centralizado de acordeones
 * Este archivo puede ser importado en cualquier página que necesite acordeones
 */

// Namespace para evitar conflictos con otras bibliotecas o scripts
const AccordionSystem = {
    // Inicializa todos los acordeones en la página
    init: function(selector = '.collapsible-fieldset', defaultCollapsed = true) {
        console.log('Inicializando sistema de acordeones');
        const fieldsets = document.querySelectorAll(selector);
        
        fieldsets.forEach(fieldset => {
            this.setupAccordion(fieldset, defaultCollapsed);
        });
        
        return fieldsets.length; // Devuelve el número de acordeones inicializados
    },
    
    // Configura un único acordeón
    setupAccordion: function(fieldset, collapsed = true) {
        // Primero, limpiar cualquier configuración previa para evitar duplicados
        this.cleanupAccordion(fieldset);
        
        // Luego configurar el estado inicial
        if (collapsed) {
            fieldset.classList.add('collapsed');
        } else {
            fieldset.classList.remove('collapsed');
        }
        
        const button = fieldset.querySelector('.collapse-toggle');
        if (button) {
            button.setAttribute('data-accordion-initialized', 'true');
            button.addEventListener('click', this.toggleAccordion);
        }
    },
    
    // Limpia la configuración anterior de un acordeón
    cleanupAccordion: function(fieldset) {
        const button = fieldset.querySelector('.collapse-toggle');
        if (button && button.getAttribute('data-accordion-initialized') === 'true') {
            // Crear un clon del botón sin event listeners
            const newButton = button.cloneNode(true);
            button.parentNode.replaceChild(newButton, button);
        }
    },
    
    // Manejador de eventos para el botón toggle
    toggleAccordion: function(event) {
        const fieldset = this.closest('.collapsible-fieldset');
        if (fieldset) {
            fieldset.classList.toggle('collapsed');
            console.log('Acordeón toggled:', fieldset);
            
            // Opcional: disparar un evento personalizado que otros scripts puedan escuchar
            const toggleEvent = new CustomEvent('accordion:toggled', { 
                detail: { 
                    fieldset: fieldset,
                    isCollapsed: fieldset.classList.contains('collapsed')
                },
                bubbles: true 
            });
            fieldset.dispatchEvent(toggleEvent);
        }
    },
    
    // Colapsa todos los acordeones
    collapseAll: function(selector = '.collapsible-fieldset') {
        document.querySelectorAll(selector).forEach(fieldset => {
            fieldset.classList.add('collapsed');
        });
    },
    
    // Expande todos los acordeones
    expandAll: function(selector = '.collapsible-fieldset') {
        document.querySelectorAll(selector).forEach(fieldset => {
            fieldset.classList.remove('collapsed');
        });
    },
    
    // Reinicializar acordeones después de cargar contenido dinámicamente
    reinitAfterContentLoad: function(container, defaultCollapsed = true) {
        if (!container) {
            console.error('Container is null or undefined');
            return 0;
        }
        
        const selector = '.collapsible-fieldset';
        const accordions = container.querySelectorAll(selector);
        console.log('Reinicializando', accordions.length, 'acordeones en', container);
        
        accordions.forEach(accordion => {
            this.setupAccordion(accordion, defaultCollapsed);
        });
        
        return accordions.length;
    }
};

// Auto-inicialización cuando el DOM está listo
document.addEventListener('DOMContentLoaded', function() {
    AccordionSystem.init();
});

// Exportar para uso en otros scripts
window.AccordionSystem = AccordionSystem;