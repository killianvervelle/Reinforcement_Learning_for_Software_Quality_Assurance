import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';

const ComparisonSupply = ({ data }) => {
    const chartRef = useRef(null);
    const [selectedYear, setSelectedYear] = useState('year_2014');

    useEffect(() => {
        if (!data || data.length === 0) return;

        const dataArray = Object.entries(data);

        d3.select(chartRef.current).selectAll('svg').remove();
        // Filter data for the year 2014
        const yearData = dataArray.map(([name, item]) => {
            const selectedData = item[selectedYear] || {};
            const simplifiedData = {
              name,
              values: selectedData,
            };
            return simplifiedData;
          });
        // Sort data by "Production" value in descending order
        yearData.sort((a, b) => (b.values["Feed"] || 0) - (a.values["Feed"] || 0));

        // Display the top 10 items
        const top10Data = yearData.slice(0, 10);
  
        // Set up D3 chart
        const margin = { top: 50, right: 20, bottom: 120, left: 50 };
        const width = 500 - margin.left - margin.right;
        const height = 400 - margin.top - margin.bottom;

        const svg = d3.select(chartRef.current)
        .append('svg')
        .attr('width', width + margin.left + margin.right)
        .attr('height', height + margin.top + margin.bottom)
        .append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

        // Set up scales
        const xScale = d3.scaleBand()
        .domain(top10Data.map(d => d.name))
        .range([0, width])
        .padding(0.1);

        const yScale = d3.scaleLinear()
        .domain([0, d3.max(top10Data, d => d3.max(Object.values(d.values)))])
        .range([height, 0]);

        // Define the custom order of keys
        const keyOrder = ["Production", "Feed", "Losses", "Seed"];

        // Draw grouped bars
        const barWidth = xScale.bandwidth() / keyOrder.length;

        top10Data.forEach((item, index) => {
        keyOrder.forEach((key, i) => {
            svg.append('rect')
            .attr('x', xScale(item.name) + i * barWidth)
            .attr('y', yScale(item.values[key] || 0))
            .attr('width', barWidth)
            .attr('height', height - yScale(item.values[key] || 0))
            .attr('fill', d3.schemeCategory10[i]);
        });
        });

        // Draw axes
        const xAxis = d3.axisBottom(xScale);
        svg.append('g')
        .attr('transform', `translate(0,${height})`)
        .call(xAxis)
        .selectAll('text')
        .attr('transform', 'rotate(-45)')
        .style('text-anchor', 'end');

        const yAxis = d3.axisLeft(yScale);
        svg.append('g')
        .call(yAxis);

        // Draw title
        svg.append('text')
        .attr('x', width / 2)
        .attr('y', -margin.top / 2)
        .attr('text-anchor', 'middle')
        .style('font-size', '16px')
        .style('font-weight', 'bold')
        .text('Top 10 Items with the Highest Feed to Production Ratio');

        // Draw x-axis label
        svg.append('text')
        .attr('x', width / 2)
        .attr('y', 330)
        .attr('text-anchor', 'middle')
        .style('font-size', '14px')
        .text('Item');

        // Draw y-axis label
        svg.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('x', -height / 2)
        .attr('y', -margin.left + 10)
        .attr('text-anchor', 'middle')
        .style('font-size', '14px')
        .text('Megatonnes');

        // Draw legend
        const legend = svg.append('g')
        .attr('transform', `translate(${width - 100}, 0)`);

        keyOrder.forEach((key, i) => {
        legend.append('rect')
            .attr('x', 0)
            .attr('y', i * 20)
            .attr('width', 10)
            .attr('height', 10)
            .attr('fill', d3.schemeCategory10[i]);

        legend.append('text')
            .attr('x', 20)
            .attr('y', i * 20 + 10)
            .text(key)
            .style('font-size', '12px')
            .attr('alignment-baseline', 'middle');
        });
    }, [data, selectedYear]);
    return (
        <div>
          <svg ref={chartRef} height={400} width={500}></svg>
          <div>
            <label>
              <input
                type="radio"
                value="year_2014"
                checked={selectedYear === 'year_2014'}
                onChange={() => setSelectedYear('year_2014')}
              />
              2014
            </label>
            <label>
              <input
                type="radio"
                value="year_2019"
                checked={selectedYear === 'year_2019'}
                onChange={() => setSelectedYear('year_2019')}
              />
              2019
            </label>
          </div>
        </div>
      );
    };
export default ComparisonSupply;
