package tritonparser

import "testing"

type testResponse struct {
	b       [][]byte
	outputs []*testResponsOutput
}

type testResponsOutput struct {
	name     string
	datatype string
	shape    []int64
}

func TestUnmarshal(t *testing.T) {
}
