package main

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

// TODO:
// 6. check that my gradients are correct

func predict(thetas []*mat.Dense, input *mat.Dense) (aVals, zVals []*mat.Dense) {
	biasedExamples := addBiasCol(input, 1)
	aVals = make([]*mat.Dense, len(thetas)) // slice for a values (after sigmoid application)
	zVals = make([]*mat.Dense, len(thetas)) // slice for z values (before sigmoid)

	var currentOut mat.Dense
	currentOut.Clone(biasedExamples) // make a copy of the biased examples
	for i := range thetas {
		var z mat.Dense
		z.Mul(&currentOut, thetas[i].T())
		zVals[i] = &z

		var a mat.Dense
		a.Apply(sigmoid, &z)
		aVals[i] = &a

		next := addBiasCol(&a, 1)
		currentOut.Clone(next) // make a new copy of the current layer output
	}

	return aVals, zVals
}

func getCostAndGradients(thetas []*mat.Dense, examples, labels *mat.Dense, lambda float64) (float64, []*mat.Dense) {
	numExamples, _ := examples.Dims()

	aValues, zValues := predict(thetas, examples)

	//for i := range thetas {
	//	fmt.Printf("thet[%d] size: ", i)
	//	printDims(thetas[i])
	//}
	//fmt.Printf("\n")

	//for i := range aValues {
	//	fmt.Printf("aValues[%d] size: ", i)
	//	printDims(aValues[i])
	//}
	//fmt.Printf("\n")

	//for i := range zValues {
	//	fmt.Printf("zValues[%d] size: ", i)
	//	printDims(zValues[i])
	//}
	//fmt.Printf("\n")

	deltas := make([]*mat.Dense, len(thetas))

	// TODO: make this into a loop
	//compute and append delta4
	var delta3 mat.Dense
	delta3.Sub(aValues[len(aValues)-1], labels)
	deltas[len(thetas)-1] = &delta3

	//fmt.Printf("delta[%d] size: ", 1)
	//printDims(deltas[1])

	//compute and append delta3, d4 * theta3 .* sigmoidGradient(z3)
	var delta2 mat.Dense
	delta2.Mul(thetas[1].T(), delta3.T())

	r, c := delta2.Dims()
	delta2 = *mat.DenseCopyOf(delta2.Slice(1, r, 0, c))
	//fmt.Printf("delta[%d] size: ", 2)
	//printDims(&delta2)
	delta2.MulElem(sigmoidGradient(mat.DenseCopyOf(zValues[0].T())), &delta2)
	deltas[len(deltas)-2] = &delta2

	//for i := range deltas {
	//	fmt.Printf("delta[%d] size: ", i)
	//	printDims(deltas[i])
	//}
	//fmt.Printf("\n")

	// TODO: add lambda normalization
	// make into a loop
	gradients := make([]*mat.Dense, len(thetas))

	var Delta1 mat.Dense
	Delta1.Mul(deltas[0], examples)
	Delta1.Apply(func(i, j int, v float64) float64 { return v / float64(numExamples) }, &Delta1)

	r, c = thetas[0].Dims() // lambda normalization
	t := mat.DenseCopyOf(thetas[0].Slice(0, r, 1, c))
	t.Apply(func(i, j int, v float64) float64 { return lambda / float64(numExamples) * v }, t)
	Delta1.Add(&Delta1, t)

	gradients[0] = &Delta1

	var Delta2 mat.Dense
	Delta2.Mul(deltas[1].T(), aValues[0])
	Delta2.Apply(func(i, j int, v float64) float64 { return v / float64(numExamples) }, &Delta2)

	r, c = thetas[1].Dims() // lambda normalization
	t = mat.DenseCopyOf(thetas[1].Slice(0, r, 1, c))
	t.Apply(func(i, j int, v float64) float64 { return lambda / float64(numExamples) * v }, t)
	Delta2.Add(&Delta2, t)
	gradients[1] = &Delta2

	//for i := range gradients {
	//	fmt.Printf("gradients[%d] size: ", i)
	//	printDims(gradients[i])
	//}
	//fmt.Printf("\n")

	J := cost(labels, aValues[len(aValues)-1])

	// lambda regularization
	//% regularize the cost function, byt adding up all of the non bias theta terms
	//r = sum(sum(Theta1 .^ 2)(:, 2:end)) + sum(sum(Theta2 .^ 2)(:, 2:end)); r = r * lambda / (2 * m); J += r;
	reg := float64(0)
	for i := range thetas {
		tmp := mat.DenseCopyOf(thetas[i])
		tmp.Apply(func(i, j int, v float64) float64 { return v * v }, tmp)
		reg += mat.Sum(tmp)
	}
	reg = reg * lambda / (2 * float64(numExamples))
	J += reg

	return J, gradients
}

// TODO: implement this
func gradientCheck() {

}

func cost(labels, aValues *mat.Dense) float64 {
	// work out the cost function
	// 1. (-exp .* log(aLast) J = -y_exp .* log(a_three) - (1 - y_exp) .* log(1 - a_three)) / m;
	var j mat.Dense
	j.Clone(labels)
	j.Apply(func(i, j int, v float64) float64 { return v * -1 }, &j)

	var r mat.Dense
	r.Clone(aValues)
	r.Apply(func(i, j int, v float64) float64 { return math.Log(v) }, &r)

	j.MulElem(&j, &r)

	// 2. j - (1 - exp) .* log(1 - aLast)
	var k mat.Dense
	k.Clone(labels)
	k.Apply(func(i, j int, v float64) float64 { return 1 - v }, &k)

	var l mat.Dense
	l.Clone(aValues)
	l.Apply(func(i, j int, v float64) float64 { return math.Log(1 - v) }, &l)

	k.MulElem(&k, &l)

	j.Sub(&j, &k)

	// 3. sum the elements
	return mat.Sum(&j)
}

func main() {
	examples, labels := getData()

	// generate both thetas without bias terms
	// ThetaOne will be a 3x5 matrix. 3 neurons that each take in 5 features
	// ThetaTwo will be a 2x3 matrix, as the second layer, it has two neurons which take 3 inputs each
	thetas := []*mat.Dense{
		generateRandomTheta(6, 3, 9, 5+1),
		generateRandomTheta(3, 1, 1, 9+1),
	}

	r, c := examples.Dims()
	trainSet := mat.DenseCopyOf(examples.Slice(0, r/10, 0, c))
	testSet := mat.DenseCopyOf(examples.Slice(r/10, r, 0, c))
	trainLabels := mat.DenseCopyOf(labels.Slice(0, r/10, 0, 1))
	testLabels := mat.DenseCopyOf(labels.Slice(r/10, r, 0, 1))

	lambda := float64(0.01)
	for i := 0; i < 1000; i++ {
		cost, gradients := getCostAndGradients(thetas, trainSet, trainLabels, lambda)
		fmt.Println(cost)

		for i := range thetas {
			var n mat.Dense

			//printDims(thetas[i])
			//printDims(gradients[i])

			gradients[i].Apply(func(i, j int, v float64) float64 { return v * 7 }, gradients[i])
			r, c := thetas[i].Dims() // slice off column
			n = *mat.DenseCopyOf(thetas[i].Slice(0, r, 1, c))

			//fmt.Printf("thetas[%d] size: ", i)
			//printDims(thetas[i])
			//fmt.Printf("gradientds[%d] size: ", i)
			//printDims(gradients[i])

			n.Sub(&n, gradients[i])

			thetas[i] = addBiasCol(&n, 1) // add column back in as ones
		}

	}

	aVals, _ := predict(thetas, testSet)
	aVals[1].Sub(aVals[1], testLabels)
	printMatrix(aVals[1], "error")
}
