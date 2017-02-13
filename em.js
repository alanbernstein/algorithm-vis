function numeric_test() {
    A = [
        [1,2,3],
        [4,5,6],
        [7,3,9]
    ];

    console.log(A);

    x = [3,1,2];

    b = numeric.dot(A,x);
    Ainv = numeric.inv(A);
    numeric.dot(Ainv,b);
    numeric.det(A);
    ev = numeric.eig(A)
    ev.lambda.x
    svd = numeric.svd(A)

    console.log(svd);
}

function mean(x) {
    // mean of an NxD vector
    var mu = []
    for(var k=0; k<x[0].length; k++) {
        var c = get_column(x, k)
        mu.push(numeric.sum(c)/c.length)
    }
    return mu
}

function covariance(x, y) {
    // covariance of two Nx1 vectors
    var mx = numeric.sum(x) / x.length
    var cx = numeric.sub(x, mx)
    var my = numeric.sum(y) / y.length
    var cy = numeric.sub(y, my)
    var covar = 0
    for(n=0; n<x.length; n++) {
        covar += cx[n]*cy[n]
    }
    return covar
}

function get_column(x, n) {
    // TODO find better way
    var c = []
    for(var k=0; k<x.length; k++) {
        c.push(x[k][n])
    }
    return c
}

function covariance_matrix(x) {
    // x is an NxD matrix
    // TODO not very efficient
    var N = x.length
    var D = x[0].length
    var S = []
    for(var i=0; i<D; i++) {
        Srow = []
        var xi = get_column(x, i)
        for(var j=0; j<D; j++) {
            var xj = get_column(x, j)
            var c = covariance(xi, xj)
            Srow.push(c)
        }
        S.push(Srow)
    }
    return S
}

function init_array(N, D, v) {
    var val = []
    for(n=0; n<N; n++) {
        var row = []
        for(d=0; d<D; d++) {
            row.push(v)
        }
        val.push(row)
    }
    return val
}

function EM(x, C, thresh) {
    estimate = EM_init(x, C)
    var n=1
    while(n < 2 || delta > thresh) {
        estimate_new = EM_core(x, estimate)
        delta = Math.abs(estimate_new.l - estimate.l)
        estimate = estimate_new
        n++
    }
}

function EM_animated(axis_id, axis_params, x, C, thresh) {
    estimate = EM_init(x, C)
    init_plots(axis_id, axis_params, estimate)
    var n=1
    while(n < 2 || delta > thresh) {
        estimate_new = EM_core(x, estimate)
        delta = Math.abs(estimate_new.l - estimate.l)
        // update plot here
        estimate = estimate_new
        n++
    }
}

function EM_init(x, C) {
    // initial estimates
    var tau_est = []
    var mu_est = []
    var S_est = []
    var m = mean(x)
    var cov = covariance_matrix(x)
    for(var n=0; n<C; n++) {
        tau_est.push(1/C)
        var r = Math.random() * 2
        mu_est.push(numeric.dot(m, r))
        S_est.push(cov)
    }

    // initial log likelihood
    var l = init_array(x.length, 1, 0)
    // TODO have not verified this works properly
    for(var n=0; n<C; n++) {
        for(var k=0; k<x.length; k++) {
            var prob = multivariate_normal_pdf(x[k], mu_est[n], S_est[n])
            var d = numeric.dot(prob, tau_est[n])
            l[k] += Math.log(d)  // why doesnt numeric work here?
        }
    }

    return {
        C: C,
        N: x.length,
        tau: tau_est,
        mu: mu_est,
        S: S_est,
        l: mean(l),
    }
}

function EM_core(x, estimate) {
    // E step
    var A = init_array(x.length, estimate.C, 0)
    for(var n=0; n<estimate.C; n++) {
        for(var k=0; k<x.length; k++) {
            var prob = multivariate_normal_pdf(x[k], estimate.mu[n], estimate.S[n])
            A[k, n] = estimate.tau[n] * prob
        }
    }

    // M step

    // compute log-likelihood, convergence check


}

function standard_normal_pdf(x) {
    return 1/Math.sqrt(2*Math.PI) * Math.exp(-x*x/2)
}

function normal_pdf(x, mu, s) {
    return standard_normal_pdf((x-mu)/s)
}

function multivariate_standard_normal_pdf(x) {
    var val = 1
    for(n=0; n<x.length; n++) {
        val *= standard_normal_pdf(x[n])
    }
    return val
}

function multivariate_normal_pdf(x, mu, S) {
    xt = numeric.sub(x, mu)
    // http://www.numericjs.com/documentation.html
    // https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Geometric_interpretation
    // TODO transform according to S
    return multivariate_standard_normal_pdf(xt)
}

function standard_normal_sample() {
    // http://stackoverflow.com/questions/25582882/javascript-math-random-normal-distribution-gaussian-bell-curve
    // Standard Normal variate using Box-Muller transform.
    // better options: marsaglia polar, ziggurat
    var u = 1 - Math.random(); // Subtraction to flip [0, 1) to (0, 1].
    var v = 1 - Math.random();
    return Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
}

function standard_normal_samples(N) {
    samples = []
    for(n=0; n<N; n++) {
        var u = 1 - Math.random(); // Subtraction to flip [0, 1) to (0, 1].
        var v = 1 - Math.random();
        samples.push(Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v ));
    }
    return samples
}


function normal_samples(N, mu, sigma) {
    samples = []
    for(n=0; n<N; n++) {
        var u = 1 - Math.random(); // Subtraction to flip [0, 1) to (0, 1].
        var v = 1 - Math.random();
        standard = Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v )
        sample = standard * sigma + mu
        samples.push(sample);
    }
    return samples
}

function multivariate_standard_normal_sample(D) {
    point = []
    for(d=0; d<D; d++) {
        var u = 1 - Math.random(); // Subtraction to flip [0, 1) to (0, 1].
        var v = 1 - Math.random();
        standard = Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
        point.push(standard)
    }
    return point
}

function multivariate_standard_normal_samples(N, D) {
    samples = []
    for(n=0; n<N; n++) {
        point = []
        for(d=0; d<D; d++) {
            var u = 1 - Math.random(); // Subtraction to flip [0, 1) to (0, 1].
            var v = 1 - Math.random();
            standard = Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
            point.push(standard)
        }
        samples.push(point)
    }
    return samples
}

function multivariate_normal_samples(N, mu, cov) {
    // TODO: one of
    // - use numeric in here
    // - implement linear algebra stuff in here from scratch
    // - just use multivariate_standard and use numeric externally
}

function MoG_samples(N, tau, mu, cov) {
    var dims = tau.length
    var standard = multivariate_standard_normal_samples(N, dims);
    var samples = []
    var classes = []
    for(var n=0; n<N; n++) {
        var c = sample_pmf(tau, [0, 1, 2]);
        var x = standard[n]
        var y = numeric.add(numeric.dot(cov[c], x), mu[c])
        samples.push(y)
        classes.push(c)
    }

    return {
        tau: tau,
        mu: mu,
        cov: cov,
        samples: samples,
        classes: classes,
        tau_emp: [],
        mu_emp: [],
        cov_emp: [],
    }
}


function sample_pmf(weights, values) {
    //weights = [0.3, 0.3, 0.3, 0.1];
    //results = [1, 2, 3, 4];
    var num = Math.random(),
        s = 0,
        lastIndex = weights.length - 1;

    for (var i = 0; i < lastIndex; ++i) {
        s += weights[i];
        if (num < s) {
            return values[i];
        }
    }
    return values[lastIndex];
}

function generate_static_data() {
    var N = 1000;

    var tau = [1/3, 1/3, 1/3],
        mu = [[1, 0], [0, 1], [1, 1]];

    var S0 = [[[.3, 0], [0, .2]], [[.1, 0], [0, .4]]],
        S = [numeric.dot(S0[0], S0[0]), numeric.dot(S0[1], S0[1])];

    var t = Math.PI/4,
        r = [[Math.cos(t), -Math.sin(t)], [Math.sin(t), Math.cos(t)]],
        s = [[.1, 0],[0, .4]]
    S[2] = numeric.dot(numeric.dot(numeric.dot(r,s), s), numeric.inv(r))

    return MoG_samples(N, tau, mu, S)
}

function generate_random_data(C, N) {
    // C = number of classes
    // N = number of points
    var D = 2
    var tau = numeric.random([C])
    tau = numeric.div(tau, numeric.sum(tau))

    var mu = []
    var S = []
    for(var c=0; c<C; c++) {
        mu.push(numeric.random([D]))

        var x = numeric.random([D, D])
        var A = numeric.neg(numeric.log(x))
        S.push(numeric.dot(A, numeric.transpose(A)))
        // S.push([[.1, 0], [0, .2]])
    }

    return MoG_samples(N, tau, mu, S)
}
