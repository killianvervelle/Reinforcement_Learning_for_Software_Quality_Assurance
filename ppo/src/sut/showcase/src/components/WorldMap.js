// components/WorldMap.js
import React, { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";

import * as d3 from 'd3';

// Styles
import '../styles/WorldMap.css';

function WorldMap() {
  const mapRef = useRef();
  const navigate = useNavigate();

  const [shouldZoom, setShouldZoom] = useState(false);
  const [selectedCountry, setSelectedCountry] = useState(null);
  const [popupPosition, setPopupPosition] = useState({ x: 0, y: 0 });
  const [popupData, setPopupData] = useState({ country: "", value: 0, values: []});
  const [data, setData] = useState([]);

  
  const zoom = d3.zoom().scaleExtent([1, 20]).on('zoom', handleZoom);

  const [selectedDropdownCountry, setSelectedDropdownCountry] = useState("");


  const handleDropdownChange = (event) => {
    const selectedCountryName = event.target.value;
    const selectedCountryInfo = Object.entries(data).find(([name, info]) => info.iso3 === selectedCountryName) || [selectedCountryName, { iso3: selectedCountryName, values: ['Undefined'] }];
    const [name, countryData] = selectedCountryInfo;

    // Calculate the center position of the SVG
    const svgRect = mapRef.current.getBoundingClientRect();
    const svgCenterX = svgRect.left + svgRect.width / 2;
    const svgCenterY = svgRect.top + svgRect.height / 2;

    // Calculate the popup dimensions
    const popupWidth = 250;
    const popupHeight = 250;

    const scrollX = window.scrollX || window.pageXOffset;
    const scrollY = window.scrollY || window.pageYOffset;

    // Calculate the position to center the popup
    const popupX = svgCenterX - popupWidth / 2 + scrollX;
    const popupY = svgCenterY - popupHeight / 2 + scrollY;


    setSelectedCountry(selectedCountryInfo[1].iso3);
    setPopupPosition({ x: popupX, y: popupY });
    setPopupData({ country: name, iso3: selectedCountryInfo[1].iso3, value: countryData.values[countryData.values.length - 1], values: countryData.values });
    setShouldZoom(false);
    setSelectedDropdownCountry(countryData.iso3);
  };

  const getCountryFeatureByName = (countryName) => {
    const worldGeojson = require('../assets/worldmap.json');
    return worldGeojson.features.find(feature => feature.properties.name === countryName);
  };

  function handleZoom(event) {
    const { transform } = event;
    const zoomLevel = transform.k;
    d3.select('svg').selectAll('path').style('stroke-width', 1 / zoomLevel).attr('transform', transform);
    setSelectedCountry(null);
  }

  const handleClick = (event, d) => {
    const countryId = d ? d.properties.adm0_a3_us : null;
    const countryName = d ? d.properties.name : null;

    if (countryId) {
      const countryInfo = Object.entries(data).find(([name, info]) => info.iso3 === countryId) || [countryName, { iso3: countryId, values: ['Undefined'] }];
      const [name, countryData] = countryInfo;

      // Calculate the center position of the SVG
      const svgRect = mapRef.current.getBoundingClientRect();
      const svgCenterX = svgRect.left + svgRect.width / 2;
      const svgCenterY = svgRect.top + svgRect.height / 2;

      // Calculate the popup dimensions
      const popupWidth = 250;
      const popupHeight = 250;

      const scrollX = window.scrollX || window.pageXOffset;
      const scrollY = window.scrollY || window.pageYOffset;

      // Calculate the position to center the popup
      const popupX = svgCenterX - popupWidth / 2 + scrollX;
      const popupY = svgCenterY - popupHeight / 2 + scrollY;

      setSelectedCountry(countryId);
      setPopupPosition({ x: popupX, y: popupY });
      setPopupData({ country: name, iso3 : countryData.iso3 , value: countryData.values[countryData.values.length -1], values: countryData.values});
      setShouldZoom(false);
    } else {
      setSelectedCountry(null);
    }
  };

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch('http://127.0.0.1:8000/undernourishement-data');
        if (!response.ok) {
          throw new Error('Failed to fetch data');
        }
        const result = await response.json();
        setData(result);
      } catch (error) {
        console.error('Error fetching data:', error);
      }
    };

    fetchData();
  }, []);

  const ChartComponent = ({ data }) => {
    const chartRef = useRef(null);
  
    useEffect(() => {
      if (data && data.length > 0) {
        d3.select(chartRef.current).selectAll('*').remove();

        const margin = { top: 20, right: 20, bottom: 20, left: 30 };
        const width = 220 - margin.left - margin.right;
        const height = 165 - margin.top - margin.bottom;
        const svg = d3.select(chartRef.current)
          .append('svg')
          .attr('width', width + margin.left + margin.right)
          .attr('height', height + margin.top + margin.bottom)
          .append('g')
          .attr('transform', `translate(${margin.left},${margin.top})`);
        
        const xLabels = ['2015', '2016', '2017', '2018', '2019', '2020'];
        const dataPoints = xLabels.map((label, i) => ({label, value: data[i]}));

        const xScale = d3.scalePoint()
          .domain(xLabels)
          .range([0, width])
          .padding(0.5);
        
        const yScale = d3.scaleLinear()
          .domain([0, d3.max(data)])
          .range([height, 0]);
        
        const line = d3.line()
          .x(d => xScale(d.label))
          .y(d => yScale(d.value));

        svg.append('path')
          .datum(dataPoints)
          .attr('fill', 'none')
          .attr('stroke', 'red')
          .attr('stroke-width', 1.5)
          .attr('d', line);

        svg.append('text')
          .attr('x', width / 2)
          .attr('y', -margin.top / 2)
          .attr('text-anchor', 'middle')
          .style('font-size', '10px')
          .style('font-weight', 'bold')
          .text('Malnutrition Rate Over 6 Years');
  
        svg.append('g')
          .attr('transform', `translate(0,${height})`)
          .call(d3.axisBottom(xScale))
  
        svg.append('g')
          .call(d3.axisLeft(yScale));
      }
    }, [data]);
  
    return <div ref={chartRef}></div>;
  };

  const drawMap = (geojson, data) => {
    const width = mapRef.current.offsetWidth;
    const height = 600;

    const svgExists = d3.select(mapRef.current).select('svg').size() > 0;

    if (svgExists) {
      d3.select(mapRef.current).select('svg').remove();
    }

    const svg = d3.select(mapRef.current)
                  .append('svg')
                  .attr('width', width)
                  .attr('height', height)
                  .style('background-color', '#B4E2FF');

    const projection = d3.geoNaturalEarth1().fitSize([width, height], geojson);
    const path = d3.geoPath().projection(projection);

    svg.call(zoom);

    const countryDataMap = new Map(
      Object.entries(data).map(([countryName, countryData]) => [
        countryData.iso3,
        countryData.values[0]
      ])
    );

    svg.selectAll('path')
       .data(geojson.features)
       .enter()
       .append('path')
       .attr('d', path)
       .style('stroke', 'white')
       .style('stroke-width', 1)
       .style('fill', d => getColor(countryDataMap.get(d.properties.adm0_a3_us)))
       .on('mouseover', handleMouseOver)
       .on('mouseout', handleMouseOut)
       .on('click', handleClick);

    function handleMouseOver(event, d) {
      const countryName = d.properties.name;
    
      // Get the current fill color
      const currentColor = d3.select(this).style('fill');
    
      // Lighten the color
      const lighterColor = d3.rgb(currentColor).brighter(0.9).toString();
    
      d3.select(this).style('fill', lighterColor);
    
      const centroid = path.centroid(d);

      // Remove existing tooltip
      svg.select('#tooltip').remove();

      // Append a new text element for the tooltip
      svg.append('text')
        .attr('id', 'tooltip')
        .attr('x', centroid[0])
        .attr('y', centroid[1])
        .attr('text-anchor', 'middle')
        .style('font-size', '14px')
        .style('font-weight', 'bold')
        .style('fill', 'black')
        .text(countryName);
    }
    

    function handleMouseOut(event, d) {
      d3.select(this).style('fill', d => getColor(countryDataMap.get(d.properties.adm0_a3_us)));

      svg.select('#tooltip').remove();
    }
  };

  const getColor = value => {
    var colorScale = d3.scaleThreshold()
    .domain([5, 15, 100])
    .range(["#03b082", "#fa7448", "#940f42"]);
    return value !== undefined ? colorScale(value) : '#b1ada4';
  };

  const handleResize = () => {
    const newWidth = window.innerWidth;
    const newHeight = 600;

    d3.select(mapRef.current)
      .select('svg')
      .attr('width', newWidth)
      .attr('height', newHeight);

    // Update the zoom buttons position
    updateZoomButtonsPosition();

    drawMap(require('../assets/worldmap.json'), data);
  };

  useEffect(() => {
    const worldGeojson = require('../assets/worldmap.json');
  
    if (!shouldZoom) {
      drawMap(worldGeojson, data);
    }
  
    
    handleResize();
    window.addEventListener('resize', handleResize);
    
    window.addEventListener('scroll', handleScroll);
    return () => {
      
      window.removeEventListener('scroll', handleScroll);
      window.removeEventListener('resize', handleResize);
    };
  }, [shouldZoom, data]);
  
  const handleZoomIn = () => {
    d3.select('svg').transition().call(zoom.scaleBy, 1.5);
  };

  const handleZoomOut = () => {
    d3.select('svg').transition().call(zoom.scaleBy, 0.5);
  };

  const handleScroll = () => {
    // Update the zoom buttons position on scroll
    updateZoomButtonsPosition();
  };

  const updateZoomButtonsPosition = () => {
    const mapRect = mapRef.current.getBoundingClientRect();
    const zoomButtons = document.querySelector('.zoom-buttons');

    // Calculate the desired position based on the top-left corner of the map container
    const top = mapRect.top + window.scrollY + 60; // Adjust the value as needed
    const left = mapRect.left + window.scrollX + 10; // Adjust the value as needed

    // Set the position of the zoom buttons
    zoomButtons.style.top = `${top}px`;
    zoomButtons.style.left = `${left}px`;
  };


  return (
    <div className="map-container" ref={mapRef}>
      <div className="dropdown-container">
        <label htmlFor="countryDropdown"><b>Search: </b></label>
        <select id="countryDropdown" value={selectedDropdownCountry} onChange={handleDropdownChange}>
          <option value="">Select a country</option>
          {Object.entries(data).map(([name, info]) => (
            <option key={info.iso3} value={info.iso3}>
              {name}
            </option>
          ))}
        </select>
      </div>
      <div className="zoom-buttons" >
        <button className="zoom-button" onClick={handleZoomIn}>+</button>
        <button className="zoom-button" onClick={handleZoomOut}>-</button>
      </div>
      {selectedCountry && (
        <div className="popup" style={{ left: popupPosition.x, top: popupPosition.y }}>
          <button className="close-button" onClick={() => setSelectedCountry(null)}>X</button>
          <h4 style={{ textAlign: 'center' }}>{popupData.country}</h4>
          <ChartComponent data={popupData.values} />
          <button onClick={() => navigate(`/country/${popupData.iso3}`)}>More</button>
        </div>
      )}
    </div>
  );
}

export default WorldMap;