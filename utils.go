package main

import (
	"fmt"
	"io/ioutil"
	"math"
	"math/rand"
	"strconv"
	"strings"

	"gonum.org/v1/gonum/mat"
)

func testExamplesAndLabelsCorrect(examples []float64, labels []float64) {
	if len(labels) != len(examples)/5 {
		panic(fmt.Sprint("len examples/5: ", len(examples), " != len labels: ", len(labels)))
	}

	for i := range labels {
		sum := 0.0
		for j := 0; j < 5; j++ {
			sum += examples[5*i+j]
		}

		if (sum > 2 && labels[i] == 0) || (sum < 2 && labels[i] == 1) {
			panic(fmt.Sprint("example: ", examples[i], " label: ", labels[i], ",  "))
		}

	}
}

func printMatrix(a mat.Matrix, title string) {
	fa := mat.Formatted(a, mat.Prefix("   "), mat.Squeeze())
	fmt.Printf("%s = %v\n\n", title, fa)
}

// getData from the csv file created with python script
func getData() (*mat.Dense, *mat.Dense) {
	dat, err := ioutil.ReadFile("./data.csv")
	if err != nil {
		panic(err)
	}

	lines := strings.Split(string(dat), "\n")
	examples := []float64{}
	labels := []float64{}
	for _, v := range lines[:len(lines)] {
		nums := strings.Split(v, " ")
		nums = append(nums[:0], nums[0:len(nums)-1]...) // slice off last empty string
		for j, s := range nums {

			n, err := strconv.Atoi(s)
			if err != nil {
				panic(err)
			}

			if j < 5 {
				examples = append(examples, float64(n))
				continue
			}

			labels = append(labels, float64(n))
		}
	}

	//slice off the last blank slice that was appended because of a newlines
	testExamplesAndLabelsCorrect(examples, labels)

	return mat.NewDense(len(labels), 5, examples), mat.NewDense(len(labels), 1, labels)
}

func printDims(a mat.Matrix) {
	r, c := a.Dims()
	fmt.Printf("rows: %d cols: %d\n", r, c)
}

// TODO: are these epsilong values generated the right way?
func generateRandomTheta(lSize, lNextSize, rows, columns int) *mat.Dense {
	epsilon := math.Sqrt(6) / math.Sqrt(float64(lSize+lNextSize))
	var v = make([]float64, rows*columns)
	for i := range v {
		v[i] = float64(-epsilon) + rand.Float64()*float64(2*epsilon)
	}

	return mat.NewDense(rows, columns, v)
}

// add a bias column to a matrix
func addBiasCol(m *mat.Dense, v float64) *mat.Dense {
	c, r := m.Dims()

	b := make([]float64, c*(r+1))
	for i := 0; i < c; i++ {
		b[i*(r+1)] = v            // set the first column in each row to 0
		oldRow := m.RawRowView(i) // get the old row
		for j := range oldRow {
			b[i*(r+1)+(j+1)] = oldRow[j] // set the new row shifted by one to the val in old row
		}
	}
	return mat.NewDense(c, r+1, b)
}

// sigmoid is the function that applies a sigmoid to every element in the matrix, the i and j terms are
// there because they are required by the gonum lib, but it is ok to input zeros otehrwize
func sigmoid(i, j int, v float64) float64 {
	return 1 / (1 + math.Exp(-v))
}

// same as sigmoid but the 1 - sigmoid for the delta terms
func oneMinusSigmoid(i, j int, v float64) float64 {
	return 1 - (1 / (1 + math.Exp(-1)))
}

// return the sigmoid gradient for a whole matrix
func sigmoidGradient(a *mat.Dense) *mat.Dense {
	var b mat.Dense
	b.Apply(sigmoid, a)

	var c mat.Dense
	c.Apply(oneMinusSigmoid, a)

	var d mat.Dense
	d.MulElem(&b, &c)

	return &d
}
