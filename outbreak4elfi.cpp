#include <boost/python/numpy.hpp>
#include "outbreak.cpp"

namespace p = boost::python;
namespace np = boost::python::numpy;
typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMatrixXi;

// simulate a batch of outbreaks each with a different R0
np::ndarray simulateR0(np::ndarray &py_R0, uint batch_size, uint seed)
{
    std::mt19937_64 prng(seed);
    params_struct params;

    // convert input R0 to Eigen
    Eigen::Map<Eigen::VectorXd> R0((double *) py_R0.get_data(), batch_size);

    // setup output matrix
    uint n_output = lrint(1. * params.max_time / params.output_interval);
    RowMatrixXi output(batch_size, n_output);

    // mean infectious period
    double mean_inf_period = params.infect_period_shape * params.infect_period_scale;

    Eigen::MatrixXi c;
    
    // loop over the batch
    for (uint i=0; i<batch_size; ++i)
    {
        // setup simulation-specific params
        params.infect_delta = mean_inf_period / R0[i];

        Outbreak ob(prng, params);
        c = ob.getCounters();
        output.row(i) = c.rowwise().sum() - c.col(0) - c.col(2);
    }

    // convert output to numpy array
    // https://github.com/boostorg/python/issues/97 -> need to copy!
    np::ndarray py_output = np::from_data(output.data(), np::dtype::get_builtin<int>(),
                                          p::make_tuple(batch_size, n_output), 
                                          p::make_tuple(sizeof(int)*n_output, sizeof(int)),
                                          p::object());

    // std::cout << "Final: " << std::endl << output << std::endl;
    // std::cout << "Py: " << p::extract<char const*>(p::str(py_output)) << std::endl;
    return py_output.copy();
}


BOOST_PYTHON_MODULE(outbreak4elfi)
{
    Py_Initialize();
    np::initialize();
    boost::python::def("simulateR0", &simulateR0);
}
