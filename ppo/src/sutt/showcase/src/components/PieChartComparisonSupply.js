import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';

const PieChartComparisonSupply = ({ data }) => {
  const chartRef = useRef(null);
  const [selectedYear, setSelectedYear] = useState('year_2014');

  useEffect(() => {
    if (!data || data.length === 0) return;

    const dataArray = Object.entries(data);

    d3.select(chartRef.current).selectAll('div').remove();

    const yearData = dataArray.map(([name, item]) => {
      const selectedData = item[selectedYear] || {};
      const simplifiedData = {
        name,
        values: selectedData,
      };
      return simplifiedData;
    });

    yearData.sort((a, b) => (b.values["Feed"] || 0) - (a.values["Feed"] || 0));

    const top5Data = yearData.slice(0, 5);

    // Set up D3 charts for pie charts
    const pieCharts = d3.select(chartRef.current)
      .selectAll('div')
      .data(top5Data)
      .enter()
      .append('div')
      .attr('class', 'pie-chart')
      .style('float', 'left')
      .style('margin', '15px')
      .each(function (d) {
        const pieSvg = d3.select(this)
          .append('svg')
          .attr('width', 165)
          .attr('height', 165)
          .append('g')
          .attr('transform', 'translate(75, 75)');

        const pie = d3.pie().value((d) => d.value);

        const arc = d3.arc().innerRadius(0).outerRadius(55);

        const color = d3.scaleOrdinal().range(d3.schemeCategory10);

        const pieData = ["Production", "Feed", "Losses", "Seed"].map(key => ({
          key,
          value: d.values[key] || 0,
        }));

        const arcs = pie(pieData);

        pieSvg.selectAll('path')
          .data(arcs)
          .enter()
          .append('path')
          .attr('d', arc)
          .attr('fill', (d) => color(d.data.key));

        // Remove labels within each pie

        // Display item name below each pie
        pieSvg.append('text')
          .attr('text-anchor', 'middle')
          .attr('dy', '1em')
          .attr('y', '55px')
          .attr('x', '7px')
          .attr('display', 'block')
          .attr('align-items', 'center')

          .text(d.name);
      });

    // Display legend below the pie charts
    const legend = d3.select(chartRef.current)
      .append('div')
      .attr('class', 'pie-legend');

    const legendItems = legend.selectAll('div')
      .data(["Production", "Feed", "Losses", "Seed"])
      .enter()
      .append('div')
      .attr('class', 'legend-item')
      .style('margin-bottom', '10px');

    legendItems.append('div')
      .attr('class', 'legend-color')
      .style('background-color', (d, i) => d3.schemeCategory10[i]);

    legendItems.append('div')
      .attr('class', 'legend-label')
      .text((d) => d);
  }, [data, selectedYear]);

  return (
    <div>
      <h2 className='title-svg'>Top 5 Items with the Highest Feed to Production Ratio</h2>
      <div ref={chartRef} style={{ display: 'flex', flexWrap: 'wrap' }}></div>
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

export default PieChartComparisonSupply;
