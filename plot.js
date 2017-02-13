// http://alignedleft.com/content/03-tutorials/01-d3/160-axes/6.html

function init_plot(padding, xmin, xmax, ymin, ymax) {
    svg = d3.select("svg"),
    width = +svg.attr("width"),
    height = +svg.attr("height");

    //Create scale functions
    var xScale = d3.scale.linear()
		    .domain([xmin, xmax])
		    .range([padding, width - padding]);

    var yScale = d3.scale.linear()
		    .domain([ymin, ymax])
		    .range([height - padding, padding]);

    //Define X axis
    var xAxis = d3.svg.axis().scale(xScale).orient("bottom").ticks(5);

    //Define Y axis
    var yAxis = d3.svg.axis().scale(yScale).orient("left").ticks(5);

    //Create X axis
    svg.append("g")
		    .attr("class", "axis")
		    .attr("transform", "translate(0," + (height - padding) + ")")
		    .call(xAxis);

    //Create Y axis
    svg.append("g")
		    .attr("class", "axis")
		    .attr("transform", "translate(" + padding + ",0)")
		    .call(yAxis);

    return {
        xScale: xScale,
        yScale: yScale
    }
}

function compute_error_ellipse(mu, cov) {
    // compute ellipse parameters from covariance matrix
    var s1 = Math.sqrt(cov[0][0])
    var s2 = Math.sqrt(cov[1][1])
    var corr = cov[0][1]
    var rho = corr/(s1*s2);
    var theta = 1/2*Math.atan(2*corr/(s1**2-s2**2));
    if(s1 == s2) {
        // not sure if this makes any sense...
        theta = 0
    }

    var c = Math.cos(theta)
    var s = Math.sin(theta);
    var num = s1**2*s2**2*(1-rho**2)
    var com = 2*rho*s1*s2*s*c
    var a = Math.sqrt(num/(s2**2*c**2 - com + s1**2*s**2)); // semi-major axis 1
    var b = Math.sqrt(num/(s2**2*s**2 + com + s1**2*c**2)); // semi-major axis 2

    var N = 64
    var data = []
    for(var n=0; n<N+1; n++) {
        t = 2 * Math.PI * n/N
        // compute (x*a + iy*b) * (c + is) + (mu[0] + i*mu[1])
        var x0 = Math.cos(t)*a
        var y0 = Math.sin(t)*b
        var x = x0 * c - y0 * s + mu[0]
        var y = x0 * s + y0 * c + mu[1]
        data.push([x, y])
    }

    return data
}

function error_ellipse(axis_id, axis_params, mu, cov) {
    var svg = d3.select("#" + axis_id)

    var data = compute_error_ellipse(mu, cov)

    var line = d3.svg.line()
        .x(function(d) { return axis_params.xScale(d[0]); })
        .y(function(d) { return axis_params.yScale(d[1]); });

    svg.append("path")
        .attr("class", "line")
        .attr("stroke", "black")
        .attr("fill", "none")
        .attr("d", line(data));
}

function scatter2(axis_id, axis_params, data) {
    var svg = d3.select("#" + axis_id)

    //Create circles
    svg.selectAll("circle")
		    .data(data)
		    .enter()
		    .append("circle")
		    .attr("cx", function(d) {
			      return axis_params.xScale(d[0]);
		    })
		    .attr("cy", function(d) {
			      return axis_params.yScale(d[1]);
		    })
		    .attr("r", 2);
}

