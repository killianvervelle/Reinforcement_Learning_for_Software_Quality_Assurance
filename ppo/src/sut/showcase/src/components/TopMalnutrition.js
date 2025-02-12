// TopMalnutritionChart.js
import React, { useEffect, useState } from 'react';

import { useNavigate } from "react-router-dom";
import * as d3 from 'd3';

import '../styles/TopMalnutrition.css'

const TopMalnutrition = ({ data, order, color, id }) => {
  const navigate = useNavigate();
  const [selectedYear, setSelectedYear] = useState(2014);

  useEffect(() => {
    // Ensure data is available
    if (!data) return;

    // Filter data for the selected year
    const yearData = Object.entries(data)
      .map(([country, { values, iso3 }]) => ({ country, malnutrition: values[selectedYear - 2014], iso3 }))
      .sort((a, b) => (order === 'asc' ? a.malnutrition - b.malnutrition : b.malnutrition - a.malnutrition))
      .slice(0, 30);

    // Create a bar chart
    const margin = { top: 50, right: 0, bottom: 10, left: 20 };
    const width = 450 - margin.left - margin.right;
    const height = 250 - margin.top - margin.bottom;

    const x = d3.scaleBand().range([0, width]).padding(0.1);
    const y = d3.scaleLinear().range([height, 0]);

    const svg = d3.select('#top-malnutrition-chart' + id);

    svg.selectAll('*').remove(); // Clear previous chart

    svg.attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    x.domain(yearData.map(d => d.country));
    y.domain([0, d3.max(yearData, d => d.malnutrition)]);

    // Rotate x-axis labels by 45 degrees
    svg.append('g')
      .attr('transform', `translate(0, ${height})`)
      .call(d3.axisBottom(x))
      .selectAll('text')
      .style('text-anchor', 'end')
      .attr('dx', '-.8em')
      .attr('dy', '.15em')
      .attr('transform', 'rotate(-45)');

    // Title
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', 0 - margin.top / 2)
      .attr('text-anchor', 'middle')
      .style('font-size', '16px')
      .style('font-weight', 'bold')
      .text('Top 30 Countries with the Highest Malnutrition Rate');

    // X-axis Label
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', height + margin.bottom / 1.5 + 120)
      .attr('text-anchor', 'middle')
      .style('font-size', '14px')
      .text('Countries');

    // Y-axis Label
    svg.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('y', 0 - margin.left - 30)
      .attr('x', 0 - height / 2)
      .attr('dy', '1em')
      .style('font-size', '14px')
      .style('text-anchor', 'middle')
      .text('Malnutrition Rate (%)');

    svg.append('g')
      .call(d3.axisLeft(y));

    svg.selectAll('.bar')
      .data(yearData)
      .enter().append('rect')
      .attr('class', 'bar')
      .attr('x', d => x(d.country))
      .attr('width', x.bandwidth())
      .attr('y', d => y(d.malnutrition))
      .attr('height', d => height - y(d.malnutrition))
      .style('fill', color)
      .on('click', function (event, d) {
        const iso3 = d.iso3;
        navigate(`/country/${iso3}`);
      });

  }, [data, selectedYear, order, color]);

  const handleYearChange = (event) => {
    setSelectedYear(parseInt(event.target.value, 10));
  };

  return (
    <div className='top-malnutrition-chart-container'>
      <svg id={'top-malnutrition-chart' + id} style={{ overflow: 'visible' }}></svg>
      <div className='slider-container'>
        <div className='slider'>
            <input
              type="range"
              min={2014}
              max={2019}
              value={selectedYear}
              onChange={handleYearChange}
            />
            <span>{selectedYear}</span>
        </div>
      </div>
    </div>
  );
};


export default TopMalnutrition;
