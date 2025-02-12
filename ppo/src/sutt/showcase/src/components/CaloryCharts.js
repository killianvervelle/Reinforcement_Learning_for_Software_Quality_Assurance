// Chart.js
import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

const Chart = ({ data }) => {
    const chartRef = useRef(null);
    const width = 400;
    const height = 350;

    
    useEffect(() => {
      if (data) {
          createChart(data);
      }
    }, [data]);

  const createChart = data => {
    // Extract data
    const years = ['2014', '2019'];
    const values = [data.kcal_per_human_2014, data.kcal_per_human_2019];
    

    // Set up the SVG container
    const svg = d3.select(chartRef.current);
    svg.selectAll('*').remove(); // Clear previous chart
    
    const margin = { top: 40, right: 40, bottom: 60, left: 60 };

    // Create scales
    const xScale = d3.scaleBand()
      .domain(years)
      .range([margin.left, width - margin.right])
      .padding(0.1);

    const yScale = d3.scaleLinear()
      .domain([0, 3500])
      .nice()
      .range([height - margin.bottom, margin.top]);

    // Create a line generator
    const line = d3.line()
      .x((d, i) => xScale(years[i]) + xScale.bandwidth() / 2)
      .y(d => yScale(d));

    // Add x-axis
    svg.append('g')
      .attr('transform', `translate(0, ${height - margin.bottom})`)
      .call(d3.axisBottom(xScale));

    // Add y-axis
    svg.append('g')
    .attr('transform', `translate(${margin.left}, 0)`)
    .call(d3.axisLeft(yScale).ticks(5).tickValues(d3.range(0, 3500, 500)));

    // Add line to the chart
    svg.append('path')
      .datum(values)
      .attr('fill', 'none')
      .attr('stroke', 'steelblue')
      .attr('stroke-width', 2)
      .attr('d', line);

    // Add x-axis label
    svg.append('text')
      .attr('x', 200)
      .attr('y', 300)
      .attr('dy', '3em')
      .attr('fill', '#000')
      .attr('font-size', 14)
      .attr('text-anchor', 'middle')
      .text('Year');

    // Add y-axis label
    svg.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('y', 55)
      .attr('x', -150)
      .attr('font-size', 14)
      .attr('dy', '-3em')
      .attr('fill', '#000')
      .attr('text-anchor', 'middle')
      .text('Average daily calorie intake (kcal)');

    // Add line to the chart
    svg.append('path')
      .datum(values)
      .attr('fill', 'none')
      .attr('stroke', 'steelblue')
      .attr('stroke-width', 2)
      .attr('d', line);

    // Add chart title
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', margin.top / 2)
      .attr('text-anchor', 'middle')
      .style('font-size', '16px')
      .style('font-weight', 'bold')
      .text('Evolution of the Average Daily Calorie Intake')
      .append('tspan')
      .attr('x', width / 2)
      .attr('dy', '1.2em') 
      .text('over 5 years');
  };

  return <svg ref={chartRef} width={width} height={height}></svg>;
};

export default Chart;
