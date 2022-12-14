let
  nixpkgs = import <nixpkgs> {};
  mach-nix = import (builtins.fetchGit {
    url = "https://github.com/DavHau/mach-nix";
    ref = "refs/tags/3.5.0";
  }) {};
  pyEnv = mach-nix.mkPython rec {

    requirements = ''
        pandas
        openpyxl
        numpy
        scipy
        mlflow
        sklearn
        scikit-image
        pillow
        matplotlib
    '';

  };
in
mach-nix.nixpkgs.mkShell {

  buildInputs = [
    pyEnv
  ] ;

  shellHook = ''
  '';
}
