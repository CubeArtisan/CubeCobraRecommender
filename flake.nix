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
          system = system;
          config.allowUnfree = true;
        };
        pythonEnv = mach-nix.lib.${system}.mkPython {
            requirements = builtins.readFile ./requirements.txt;
            # providers.tensorflow = "nixpkgs";
            # _.tensorflow.buildInputs.add = cudapaths;
          };
      in
        {
          devShell = pkgs.mkShell {
            buildInputs = [
              pythonEnv
              pkgs.cudnn_cudatoolkit_11_0
              pkgs.cudaPackages.cudatoolkit_11_0
              pkgs.linuxPackages.nvidia_x11
            ];

            shellHook = ''
              export LD_LIBRARY_PATH=${pkgs.cudatoolkit_11_0}/lib:${pkgs.cudnn_cudatoolkit_11_0}/lib:${pkgs.cudatoolkit_11_0.lib}/lib:${pkgs.linuxPackages.nvidia_x11}/lib:$LD_LIBRARY_PATH
            '';
          };
        }
      );
}
