import React, { useEffect, useState, useRef } from 'react';
import { useParams, useLocation } from 'react-router-dom';

import * as d3 from 'd3';

import '../App.css';
import '../styles/Country.css'
import "../../node_modules/flag-icons/css/flag-icons.min.css";

export default function Country() {

  const [countryData, setCountryData] = useState(null);
  const [foodToTotalRatio, setFoodToTotalRatio] = useState(null);
  const [malnutritionRates, setMalnutritionData] = useState({"country": 0});
  const { id } = useParams();
  let hoveredInfo = null;
  let categorysums = {}

  const { pathname } = useLocation();

  useEffect(() => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }, [pathname]);

  useEffect(() => {
    const fetchMalnutritionData = async () => {
      try {
        const response = await fetch(`http://127.0.0.1:8000/undernourishement-data-country/${id}`, { method: 'GET'} );
        if (!response.ok) {
          throw new Error('Failed to fetch data');
        }
        const data = await response.json();
        setMalnutritionData(data);
      } catch (error) {
        console.error('Error fetching data:', error);
      }
    };

    fetchMalnutritionData();
  }, [id]);

  useEffect(() => {
    const fetchCountryData = async () => {
      try {
        const response = await fetch(`http://127.0.0.1:8000/nutritional-data-country/${id}`, { method: 'GET'} );
        if (response.ok) {
          const data = await response.json();
          setCountryData(data);
        } else {
          console.error('Profile data not found');
          setCountryData(null);
        }
      } catch (error) {
        console.error('Error fetching profile data:', error);
        setCountryData(null);
      }
    };

    fetchCountryData();
  }, [id]);
  
  const fetchUtilizationData = async (category) => {
    try {
      const response = await fetch(`http://127.0.0.1:8000/utilization-data/${id}/${category}`, { method: 'GET'} );
      if (response.ok) {
        const data = await response.json();
        const parsedData = JSON.parse(data.data)
        const tableContainer = d3.select('#table-container');
        tableContainer.html('');
        const table = tabulate(parsedData, ["item", "value"]);
        tableContainer.node().appendChild(table.node());
      } else {
        console.error('Profile data not found');
      }
    } catch (error) {
      console.error('Error fetching profile data:', error);
    }
  };

  function tabulate(data, columns) {
    var title = `${data[0].element} in ${data[0].year} in ${data[0].unit}`;
    const tableContainer = d3.select('#table-container');
    tableContainer.html('');
    tableContainer.append('div')
        .attr('class', 'table-title')
        .text(title);
    const table = tableContainer.append('table').attr('class', 'table-body');
    const thead = table.append('thead').attr('class', 'table-header'); 
    const tbody = table.append('tbody');

    thead.append('tr')
        .selectAll('th')
        .data(columns).enter()
        .append('th')
        .text(function (column) { return column; })
        .style('width', function (column) {
            return column === 'item' ? '400px' : '100px';
         
        });

    const rows = tbody.selectAll('tr')
        .data(data)
        .enter()
        .append('tr');

    rows.selectAll('td')
        .data(function (row) {
            return columns.map(function (column) {
                return { column: column, value: row[column] };
            });
        })
        .enter()
        .append('td')
        .attr('class', function (d) {
          return d.column === 'value' ? 'value-cell' : '';
        })
        .text(function (d) {
            return d.column === 'value' ? parseFloat(d.value).toFixed(3) : d.value;
        })
        .style('text-align', function (d) {
            return d.column === 'value' ? 'center' : '';
        });

    return table;
  }

  function calculateCategorySums(data) {
    const categorySums = {};
    if (data && data.country) {
      const categories = Object.keys(data.country).slice(4);

      categories.forEach(category => {
        const categoryData = data.country[category];

        if (Array.isArray(categoryData)) {
          categoryData.forEach(item => {
            const categoryValue = item[1];
            if (!categorySums[category]) {
              categorySums[category] = categoryValue;
            } else {
              categorySums[category] += categoryValue;
            }
          });
        } else {
          console.log(`Invalid data structure for category: ${category}`);
        }
      });
    } else {
      console.log("Invalid data structure. Missing 'country' property or data is null/undefined.");
    }
    const csvHeaders = ["group", "Production", "Import Quantity", "Stock Variation", "Export Quantity", "Feed", "Seed", "Losses", "Food"];
    const csvRowInput = ["Available food", categorySums.production, categorySums.import_quantity, categorySums.stock_variation, 0, 0, 0, 0, 0];
    const csvRowOutput = ["Consumed food", 0, 0, 0, categorySums.export_quantity, categorySums.feed, categorySums.seed, categorySums.losses, categorySums.food];
    const csvContent = [csvHeaders.join(",")].concat([csvRowInput.join(","), csvRowOutput.join(",")]).join("\n");
    categorysums = categorySums;
    return csvContent;
  };

  const result = calculateCategorySums(countryData);

  const calculateFoodRatio = (categorysums) => {
    if (categorysums) {
      const ratio = categorysums.food / (categorysums.production + categorysums.import_quantity + categorysums.stock_variation);
      setFoodToTotalRatio(ratio);
    }
  };
  
  useEffect(() => {
    calculateFoodRatio(categorysums);
  });

  const ratioStyles = {
    color: foodToTotalRatio !== null ? (foodToTotalRatio < 1 ? 'red' : 'green') : 'black',
    fontSize: '16px',
    fontWeight: 'bold'
  };

  const countryName = Object.keys(malnutritionRates);

  const strengthSentence =
  countryName && foodToTotalRatio !== null
  ? foodToTotalRatio < 1 && malnutritionRates[countryName] < 5
    ? `In ${countryName}, small positive adjustments to the share of food utilized to feed the population could bring the malnutrition rate (${malnutritionRates[countryName].toFixed(1)}%) down to 0. This could be achieved by lowering the share of food used to feed animals or to seed in agriculture.`
    : foodToTotalRatio < 1 && malnutritionRates[countryName] > 5
    ? `In ${countryName}, insufficient food is being utilized to adequately feed the population, resulting in a limited food intake per individual and malnutrition (${malnutritionRates[countryName].toFixed(1)}%). Drastic measures need to be taken.`
    : foodToTotalRatio > 1 && malnutritionRates[countryName] === 0.0
    ? `In ${countryName}, enough food is being utilized to adequately feed the population. Malnutrition is close to nonexistent.`
    : 'Calculating...'
  : 'Calculating...';
  
  const ChartComponent = ({ data }) => {
    const chartRef = useRef(null);

    useEffect(() => {
        if (data && Object.keys(data).length > 0) {
            d3.select(chartRef.current).selectAll('*').remove();
            const parsedData = d3.csvParse(data);
            var subgroups = parsedData.columns.slice(1);
            var groups = parsedData.map(row => row.group);
            var margin = { top: 45, right: 130, bottom: 20, left: 120 },
                width = 600 - margin.left - margin.right,
                height = 650 - margin.top - margin.bottom;
            var maxY = d3.max(parsedData, d => d3.sum(subgroups, key => +d[key]));
            var y = d3.scaleLinear().domain([0, maxY]).range([height, 0]);

            var svg = d3.select(chartRef.current)
                .append("svg")
                .attr("width", width + margin.left + margin.right)
                .attr("height", height + margin.top + margin.bottom)
                .append("g")
                .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

            var x = d3.scaleBand()
                .domain(groups)
                .range([0, width])
                .padding([0.2]);

            svg.append("g")
                .attr("transform", "translate(0," + height + ")")
                .call(d3.axisBottom(x).tickSizeOuter(3))
                .selectAll("text")  
                .style("font-size", "14px"); 

            svg.append("g")
                .call(d3.axisLeft(y));
            
            svg.append("text")
                .attr("x", width / 2)
                .attr("y", 0 - margin.top / 2)
                .attr("text-anchor", "middle")
                .style("font-size", "18px")
                .style("font-weight", "bold")
                .text("Food Utilization in Megatonnes (2019)");
            
            svg.append("text")
                .attr("transform", "rotate(-90)")
                .attr("y", 0 - margin.left)
                .attr("x", 0 - height / 2)
                .attr("dy", "1em")
                .style("text-anchor", "middle")
                .text("Weight in megatonnes (Mt)")
                .style("font-size", "16px");

            var color = d3.scaleOrdinal()
                .domain(subgroups)
                .range(['#ff7f0e', '#2ca02c', '#377eb8', '#8c564b', '#e377c2', '#ffbb78', '#7f7f7f', '#17becf']);

            var stackedData = d3.stack().keys(subgroups)(parsedData);

            svg.append("g")
                .selectAll("g")
                .data(stackedData)
                .enter().append("g")
                .attr("fill", function (d) { return color(d.key); })
                .selectAll("rect")
                .data(function (d) { return d; })
                .enter().append("rect")
                .attr("x", function (d) { return x(d.data.group); })
                .attr("y", function (d) { return y(d[1]); })
                .attr("height", function (d) { return y(d[0]) - y(d[1]); })
                .attr("width", x.bandwidth())
                .on('mouseover', function (event, d) {
                  d3.select(this)
                      .transition()
                      .duration(100)
                      .attr('opacity', 0.7);

                  const columnName = d3.select(this.parentNode).datum().key
                  const value = d.data[columnName];

                  const numericValue = parseFloat(value).toFixed(2) || 0;

                  const text = `${columnName}: ${numericValue}`;

                  const xPosition = x(d.data.group) + x.bandwidth() / 2;
                  const yPosition = (y(d[0]) + y(d[1])) / 2;

                  hoveredInfo = { columnName, numericValue, xPosition, yPosition };

                  svg.append('text')
                      .attr('class', 'value-label')
                      .attr('x', xPosition)
                      .attr('y', yPosition)
                      .attr('text-anchor', 'middle')
                      .text(text);
              })
              .on('mouseout', function () {
                  d3.select(this)
                      .transition()
                      .duration(100)
                      .attr('opacity', 1);
          
                  svg.select('.value-label').remove();
                  hoveredInfo = null;
              })
              .on('click', function () {
                if (hoveredInfo) {
                    fetchUtilizationData(hoveredInfo.columnName);
                }
            });

            const legendSet1 = svg.selectAll('.legendSet1')
                .data(subgroups)
                .enter()
                .append('g')
                .attr('class', 'legendSet1')
                .attr('transform', (d, i) => `translate(${width - 20},${i * 22 + (i >= 3 ? 20 : 0)})`);
    
            legendSet1.append('rect')
                .attr('width', 15)
                .attr('height', 15)
                .attr('fill', d => color(d));
      
            legendSet1.append('text')
                .attr('x', 30)
                .attr('y', 9)
                .attr('dy', '.35em')
                .style('text-anchor', 'start')
                .text(d => d);
            }

    }, [data]);

    return <div ref={chartRef}></div>;
};

  return (
    <div className="page-container">
      {countryData ? (
        <div>
          <div className="country-container">
            <span className={`fi fi-${countryData.country.iso2.toLowerCase()}`}></span>
            <h1 className="country-name">{countryData.country.name}</h1>
          </div>
          <div className="parent-container">
            <div className="child-container">
              <div className="top-left">
                <ChartComponent data={result} />
              </div>
            </div>
            <div className="grid-item right-container">
              <div className="child-container top-right">
                <div id="table-container" style={{ maxHeight: '300px', overflowY: 'auto' }}>
                  <div className='no-data'>Click on any element from the stacked bar chart</div>
                </div>
              </div>
              <div className="child-container bottom-right">
                <p>
                  <b>Analysis</b>
                  <br/><br/>
                  The total amount of food available = Production + Import quantity + Stock Variation.
                  <br/>
                  But how much of this amount is really fed to the population ?
                  <br/>
                  Food / Total amount of food available =&nbsp;
                  <span style={ratioStyles}>
                    {foodToTotalRatio !== null ? (foodToTotalRatio.toFixed(1) * 100).toFixed(1) + "%" : 'Calculating...'}
                  </span>
                  <br/><br/>
                  {strengthSentence}
                </p>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="loading-container">
          <p>Loading...</p>
        </div>
      )}
    </div>
  );
}