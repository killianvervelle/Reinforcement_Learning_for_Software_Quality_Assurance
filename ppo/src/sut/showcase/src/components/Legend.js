import React from 'react';

import '../App.css';

import '../styles/Legends.css'

function Legend() {
  return (
    <>
    <div className="legend">
        <div className="legend-item">
            <div className="legend-value">Malnutrition rate</div>
        </div>
        <div className="legend-item">
            <div className="legend-color" style={{ backgroundColor: '#03b082' }}></div>
            <div className="legend-value">&lt; 5%</div>
        </div>
        <div className="legend-item">
          <div className="legend-color" style={{ backgroundColor: '#fa7448' }}></div>
          <div className="legend-value">5-15%</div>
        </div>
        <div className="legend-item">
          <div className="legend-color" style={{ backgroundColor: '#940f42' }}></div>
          <div className="legend-value">&gt; 15%</div>
        </div>
      </div>
    </>
  );
}

export default Legend;