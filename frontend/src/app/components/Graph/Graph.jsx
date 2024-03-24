import { useMemo, useRef, useEffect, useState } from "react";
import { Tooltip } from "./Tooltip";

import * as d3 from "d3";
import axios from 'axios';


export function Graph({setCurrComp, handleOpen}) {
  const [embs, setEmbs] = useState([]);
  const [clusters, setClusters] = useState({});
  useEffect(() => {
    axios.get("http://127.0.0.1:5000/get_embedding_map").then((response) => {
      setEmbs(response.data.companies);
      setClusters(response.data.clusters)
    }).catch(err => console.log(err));
  }, [])
  return <EmbedGraph embs={embs} clusters={clusters} setCurrComp={setCurrComp} handleOpen={handleOpen} />
}

const getX = (course) => {
  return course.projection[0];
};

const getY = (course) => {
  return course.projection[1];
};

const MARGIN = { top: 50, right: 50, bottom: 60, left: 100 };
const BUBBLE_MIN_SIZE = 1;
const BUBBLE_MAX_SIZE = 7;

function EmbedGraph({ embs, clusters, setCurrComp, handleOpen }) {
  const [hovered, setHovered] = useState(null);

  const COLOR_LIST = [
    '#154428', '#979af7', '#b95cdb', '#30cdd3', '#e06c71',
    '#d975c8', '#99c5d2', '#31509d', '#d7cf61', '#aae69c',
    '#7859f7', '#27df99', '#c8f28c', '#28fc58', '#3be541',
    '#9ecba1', '#542836', '#f47957', '#f207f4', '#1ff758',
    '#a37c20', '#a957bc', '#6e6d59', '#f7c8a9', '#f7a9c8'
  ];

  const width = 800;
  const height = 450;
  const axesRef = useRef(null);
  const boundsWidth = width - MARGIN.right - MARGIN.left;
  const boundsHeight = height - MARGIN.top - MARGIN.bottom;

  // Scales
  const yScale = useMemo(() => {
    const [min, max] = d3.extent(embs.map((d) => getY(d)));
    return d3.scaleLinear().domain([min, max]).range([boundsHeight, 0]).nice();
  }, [embs, height]);

  const xScale = useMemo(() => {
    const [min, max] = d3.extent(embs.map((d) => getX(d)));
    return d3.scaleLinear().domain([min, max]).range([0, boundsWidth]).nice();
  }, [embs, width]);

  const groups = embs
    .map((d) => d.label)
    .filter((x, i, a) => a.indexOf(x) == i);

  const colorScale = d3
    .scaleOrdinal()
    .domain(groups)
    .range(COLOR_LIST);

  // Render the X and Y axis using d3.js, not react
  useEffect(() => {
    const svgElement = d3.select(axesRef.current);
    svgElement.selectAll("*").remove();

    const xAxisGenerator = d3.axisBottom(xScale);
    svgElement
      .append("g")
      .attr("class", "x axis")
      .attr("transform", "translate(0," + (boundsHeight + 20) + ")")
      .call(xAxisGenerator);

    const yAxisGenerator = d3.axisLeft(yScale);
    svgElement
      .append("g")
      .attr("class", "y axis")
      .attr("transform", "translate(" + -20 + ",0)")
      .call(yAxisGenerator);

    d3.selectAll(".x.axis").style("color", "white");
    d3.selectAll(".y.axis").style("color", "white");
  }, [xScale, yScale, boundsHeight, boundsWidth]);

  // Build the shapes
  const allShapes = embs.map((d, i) => {
    return (
      <circle
        key={i}
        r={3}
        cx={xScale(getX(d))}
        cy={yScale(getY(d))}
        opacity={1}
        stroke={colorScale(d.label)}
        fill={colorScale(d.label)}
        fillOpacity={0.4}
        strokeWidth={1}
        onMouseEnter={() =>
          setHovered({
            xPos: xScale(getX(d)),
            yPos: yScale(getY(d)),
            name: d.title,
            obj: d,
            label: d.label
          })
        }
        onClick={() => {
          setCurrComp(d);
          handleOpen();
        }}
        onMouseLeave={() => {
          setHovered(null);
        }}
        style={{
          cursor:"pointer"
        }}
      />
    );
  });

  const allClusterLabels = Object.keys(clusters).map((key, idx) => {
    return (
      <text
        x={xScale(clusters[key]['center'][0])}
        y={yScale(clusters[key]['center'][1])}
        fontFamily="sans-serif"
        fontSize={7}
        color="white"
        fontWeight={300}
      >
        {clusters[key]['summary']}
      </text>
    );
  });

  return (
    <div>
      <>
        <label
          class="inline-block pl-[0.15rem] hover:cursor-pointer"
          for="flexSwitch"
        >
        </label>
      </>
      <svg width={width} height={height}>
        <g
          width={boundsWidth}
          height={boundsHeight}
          transform={`translate(${[MARGIN.left, MARGIN.top].join(",")})`}
        >
          {allShapes}
        </g>
        {/* <g>
          {allClusterLabels}
        </g> */}
        <g
          width={boundsWidth}
          height={boundsHeight}
          ref={axesRef}
          transform={`translate(${[MARGIN.left, MARGIN.top].join(",")})`}
        />
      </svg>
      <div
        style={{
          width: boundsWidth,
          height: boundsHeight,
          position: "absolute",
          top: 0,
          left: 0,
          pointerEvents: "none",
          marginLeft: MARGIN.left,
          marginTop: MARGIN.top,
        }}
      >
        <Tooltip
          interactionData={hovered}
          embs={embs}
          clusters={clusters}
        />
      </div>
    </div>
  );
}
