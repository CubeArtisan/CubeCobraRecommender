{
  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    mach_nix = {
      url = "github:DavHau/mach-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
  };

  outputs = { nixpkgs, mach-nix, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem
      (system:
      let
        pkgs = import "${nixpkgs}" {
          inherit system;
          config.allowUnfree = true;
        };
        venvDir = "./.venv2";
        defaultShellPath = pkgs.lib.makeBinPath [ pkgs.bash pkgs.coreutils pkgs.findutils pkgs.gnugrep pkgs.gnused pkgs.which ];
        cacheRequirements = pkgs.lib.readFile ./requirements.txt;
      in
        {
          devShell = pkgs.mkShell {
            name = "CubeCobraRecommender";
            buildInputs = [
              pkgs.stdenv.cc.cc.lib
              pkgs.cudnn_cudatoolkit_11_0
              pkgs.cudaPackages.cudatoolkit_11_0
              pkgs.linuxPackages.nvidia_x11
              pkgs.python38Packages.python
            ];

            shellHook = ''
              export LD_LIBRARY_PATH=${pkgs.cudatoolkit_11_0}/lib:${pkgs.cudnn_cudatoolkit_11_0}/lib:${pkgs.cudatoolkit_11_0.lib}/lib:${pkgs.linuxPackages.nvidia_x11}/lib:${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
              export CC=/usr/bin/gcc
              SOURCE_DATE_EPOCH=$(date +%s)

              if [ ! -d "${venvDir}" ]; then
                echo "Creating new venv environment in path: '${venvDir}'"
                # Note that the module venv was only introduced in python 3, so for 2.7
                # this needs to be replaced with a call to virtualenv
                ${pkgs.python38Packages.python.interpreter} -m venv "${venvDir}"
                pip install --upgrade wheel setuptools pip
              fi

              # Under some circumstances it might be necessary to add your virtual
              # environment to PYTHONPATH, which you can do here too;
              # export PYTHONPATH=$PWD/${venvDir}/${pkgs.python38Packages.python.sitePackages}/:$PYTHONPATH

              source "${venvDir}/bin/activate"

              unset SOURCE_DATE_EPOCH
            '';
          };
        }
      );
}
