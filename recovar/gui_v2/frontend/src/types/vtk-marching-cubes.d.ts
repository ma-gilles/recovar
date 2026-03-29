/**
 * Type declarations for @kitware/vtk.js/Filters/General/ImageMarchingCubes.
 *
 * The vtk.js package ships without .d.ts for this module, so we provide
 * the minimal interface needed by VtkViewer.
 */
declare module "@kitware/vtk.js/Filters/General/ImageMarchingCubes" {
  import { vtkAlgorithm } from "@kitware/vtk.js/interfaces";

  export interface IImageMarchingCubesInitialValues {
    contourValue?: number;
    computeNormals?: boolean;
    mergePoints?: boolean;
  }

  export interface vtkImageMarchingCubes extends vtkAlgorithm {
    setContourValue(value: number): boolean;
    getContourValue(): number;
    setComputeNormals(flag: boolean): boolean;
    setMergePoints(flag: boolean): boolean;
    setInputData(data: unknown): void;
    getOutputPort(): vtkPipelineConnection;
    update(): void;
    delete(): void;
  }

  export function newInstance(
    initialValues?: IImageMarchingCubesInitialValues
  ): vtkImageMarchingCubes;

  export function extend(
    publicAPI: object,
    model: object,
    initialValues?: IImageMarchingCubesInitialValues
  ): void;

  declare const vtkImageMarchingCubes: {
    newInstance: typeof newInstance;
    extend: typeof extend;
  };
  export default vtkImageMarchingCubes;
}
