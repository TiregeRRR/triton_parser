package tritonparser

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"reflect"
)

const tag = "triton"

type TritonModelInferResponse[T TritonModelInferResponseOutputs] interface {
	GetRawOutputContents() [][]byte
	GetOutputs() []T
}

type TritonModelInferResponseOutputs interface {
	GetName() string
	GetDatatype() string
	GetShape() []int64
}

// Unmarshal function is reading data from ModelInferResponse and stores values v.
// v must be pointer to structure.
// Compatibility between different versions of api should be granted by use of interfaces.
func Unmarshal[T TritonModelInferResponseOutputs](inferResponse TritonModelInferResponse[T], v any) error {
	rv := reflect.ValueOf(v)
	if rv.Kind() != reflect.Pointer || rv.IsNil() {
		return errors.New("v must be pointer")
	}

	if rv.Elem().Kind() != reflect.Struct {
		return errors.New("v must be struct")
	}

	if err := unmarshal(inferResponse, rv); err != nil {
		return err
	}

	return nil
}

func unmarshal[T TritonModelInferResponseOutputs](inferResponse TritonModelInferResponse[T], rv reflect.Value) error {
	outputs := inferResponse.GetOutputs()
	rawBytes := inferResponse.GetRawOutputContents()
	m := getTagFieldMap(rv)

	for i, o := range outputs {
		if _, ok := m[o.GetName()]; !ok {
			continue
		}

		if err := parse(m, o, rawBytes[i]); err != nil {
			return err
		}
	}

	return nil
}

func parse(fieldMap map[string]reflect.Value, output TritonModelInferResponseOutputs, rawBytes []byte) error {
	var err error
	shape := output.GetShape()

	if shape[0] > 1 {
		return errors.New("shape > 1 is not yet supported")
	}

	if len(shape) > 2 {
		return errors.New("len(shape) > 2 is not yet supported")
	}

	if len(shape) > 1 {
		err = parseToArray(fieldMap, output, rawBytes)
	} else {
		err = parseToValue(fieldMap, output, rawBytes)
	}

	if err != nil {
		return err
	}

	return nil
}

// currently cannot store function without instantiation
//
//nolint:dupl // different functions for arrays and value.
func parseToArray(
	fieldMap map[string]reflect.Value,
	output TritonModelInferResponseOutputs,
	rawBytes []byte,
	// isArray bool,
) error {
	var err error
	switch output.GetDatatype() {
	case BOOL:
		err = unmarshalArray[bool](fieldMap, output, rawBytes)
	case UINT8:
		err = unmarshalArray[uint8](fieldMap, output, rawBytes)
	case UINT16:
		err = unmarshalArray[uint16](fieldMap, output, rawBytes)
	case UINT32:
		err = unmarshalArray[uint32](fieldMap, output, rawBytes)
	case INT8:
		err = unmarshalArray[int8](fieldMap, output, rawBytes)
	case INT16:
		err = unmarshalArray[int16](fieldMap, output, rawBytes)
	case INT32:
		err = unmarshalArray[int32](fieldMap, output, rawBytes)
	case INT64:
		err = unmarshalArray[int64](fieldMap, output, rawBytes)
	case FLOAT16:
		err = fmt.Errorf("%s not yet supported", FLOAT16)
	case FLOAT32:
		err = unmarshalArray[float32](fieldMap, output, rawBytes)
	case FLOAT64:
		err = unmarshalArray[float64](fieldMap, output, rawBytes)
	case STRING:
		err = fmt.Errorf("%s not yet supported", STRING)
	default:
		return fmt.Errorf("unkwnow type: %s", output.GetDatatype())
	}

	if err != nil {
		return err
	}

	return nil
}

// currently cannot store function without instantiation
//
//nolint:dupl // different functions for arrays and value.
func parseToValue(
	fieldMap map[string]reflect.Value,
	output TritonModelInferResponseOutputs,
	rawBytes []byte,
) error {
	var err error
	switch output.GetDatatype() {
	case BOOL:
		err = unmarshalValue[bool](fieldMap, output, rawBytes)
	case UINT8:
		err = unmarshalValue[uint8](fieldMap, output, rawBytes)
	case UINT16:
		err = unmarshalValue[uint16](fieldMap, output, rawBytes)
	case UINT32:
		err = unmarshalValue[uint32](fieldMap, output, rawBytes)
	case INT8:
		err = unmarshalValue[int8](fieldMap, output, rawBytes)
	case INT16:
		err = unmarshalValue[int16](fieldMap, output, rawBytes)
	case INT32:
		err = unmarshalValue[int32](fieldMap, output, rawBytes)
	case INT64:
		err = unmarshalValue[int64](fieldMap, output, rawBytes)
	case FLOAT16:
		err = fmt.Errorf("%s not yet supported", FLOAT16)
	case FLOAT32:
		err = unmarshalValue[float32](fieldMap, output, rawBytes)
	case FLOAT64:
		err = unmarshalValue[float64](fieldMap, output, rawBytes)
	case STRING:
		err = fmt.Errorf("%s not yet supported", STRING)
	default:
		return fmt.Errorf("unkwnow type: %s", output.GetDatatype())
	}

	if err != nil {
		return err
	}

	return nil
}

func unmarshalValue[T any](
	fieldMap map[string]reflect.Value,
	resp TritonModelInferResponseOutputs,
	rawBytes []byte,
) error {
	var val T
	if fieldMap[resp.GetName()].Type() != reflect.TypeOf(val) {
		return fmt.Errorf("types doesn't match exp: %T got: %s", val, fieldMap[resp.GetName()].Type().String())
	}

	buf := bytes.NewBuffer(rawBytes)
	if err := binary.Read(buf, binary.LittleEndian, &val); err != nil {
		return fmt.Errorf("binary read failed: %w", err)
	}

	if v, ok := fieldMap[resp.GetName()]; ok {
		v.Set(reflect.ValueOf(val))
	}

	return nil
}

func unmarshalArray[T any](
	fieldMap map[string]reflect.Value,
	resp TritonModelInferResponseOutputs,
	rawBytes []byte,
) error {
	arrLen := len(resp.GetShape())
	arr := make([]T, 0, arrLen)
	if fieldMap[resp.GetName()].Type() != reflect.TypeOf(arr) {
		return fmt.Errorf("types doesn't match exp: %T got: %s", arr, fieldMap[resp.GetName()].Type().String())
	}

	arr, err := bytesToArray[T](rawBytes, arr)
	if err != nil {
		return err
	}

	if v, ok := fieldMap[resp.GetName()]; ok {
		v.Set(reflect.ValueOf(arr))
	}

	return nil
}

func bytesToArray[T any](b []byte, arr []T) ([]T, error) {
	buf := bytes.NewReader(b)
	var t T
	size := reflect.TypeOf(t).Size()
	for i := 0; i < len(b); i += int(size) {
		err := binary.Read(buf, binary.LittleEndian, &t)
		if err != nil {
			return nil, fmt.Errorf("binary read failed: %w", err)
		}

		arr = append(arr, t)
	}

	return arr, nil
}

func getTagFieldMap(rv reflect.Value) map[string]reflect.Value {
	fieldsNum := rv.Elem().NumField()
	m := make(map[string]reflect.Value)

	for i := 0; i < fieldsNum; i++ {
		field := rv.Elem().Type().Field(i).Tag.Get(tag)
		m[field] = rv.Elem().Field(i)
	}

	return m
}
