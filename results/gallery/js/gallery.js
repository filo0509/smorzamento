
// Filtering functionality
document.getElementById('filterLinear').addEventListener('change', updateFilters);
document.getElementById('filterQuadratic').addEventListener('change', updateFilters);

function updateFilters() {
    const showLinear = document.getElementById('filterLinear').checked;
    const showQuadratic = document.getElementById('filterQuadratic').checked;
    
    document.querySelectorAll('.plot-item').forEach(item => {
        const type = item.getAttribute('data-type');
        
        if ((type === 'linear' && showLinear) || (type === 'quadratic' && showQuadratic)) {
            item.style.display = '';
        } else {
            item.style.display = 'none';
        }
    });
}
