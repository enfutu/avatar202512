// Made with Amplify Shader Editor v1.9.3.3
// Available at the Unity Asset Store - http://u3d.as/y3X 
Shader "FATALERROR/iridescent neon"
{
	Properties
	{
		_Float0("Float 0", Float) = 0
		_Float17("Float 2", Float) = 0
		_Float2("Float 2", Float) = 0
		_Float3("Float 3", Range( 0 , 1)) = 0
		_NoiseSpeed("Noise Speed", Float) = 0
		_Float5("Float 5", Float) = 0
		_Float7("Float 7", Range( 0 , 1)) = 0
		_NoiseSmoothStep("Noise SmoothStep", Range( 0 , 1)) = 0.54
		_Float9("Float 9", Float) = 0
		_TextureSample0("Texture Sample 0", 2D) = "white" {}
		_Float10("Float 10", Float) = 0
		_NoiseScale("Noise Scale", Float) = 0
		_Float13("Float 13", Float) = 0
		_Float14("Float 13", Float) = 0
		_Float15("Float 15", Float) = 0
		_Float16("Float 16", Range( 0 , 1)) = 0
		_Float8("Float 8", Float) = 2
		[HideInInspector] __dirty( "", Int ) = 1
	}

	SubShader
	{
		Tags{ "RenderType" = "Opaque"  "Queue" = "Geometry+0" "IgnoreProjector" = "True" "IsEmissive" = "true"  }
		Cull Back
		CGINCLUDE
// Upgrade NOTE: excluded shader from OpenGL ES 2.0 because it uses non-square matrices
#pragma exclude_renderers gles
		#include "UnityShaderVariables.cginc"
		#include "UnityCG.cginc"
		#include "UnityPBSLighting.cginc"
		#include "Lighting.cginc"
		#pragma target 5.0
		#pragma instancing_options procedural:vertInstancingSetup
		#define UNITY_PARTICLE_INSTANCE_DATA MyParticleInstanceData
		struct MyParticleInstanceData{float3x4 transform;uint color;float animFrame;float3 noise;};
		#include "UnityStandardParticleInstancing.cginc"
		struct Input
		{
			float3 worldNormal;
			float3 worldPos;
		};

		uniform float _NoiseSmoothStep;
		uniform float _NoiseSpeed;
		uniform float _NoiseScale;
		uniform float _Float0;
		uniform float _Float5;
		uniform sampler2D _TextureSample0;
		uniform float _Float10;
		uniform float _Float17;
		uniform float _Float3;
		uniform float _Float16;
		uniform float _Float2;
		uniform float _Float13;
		uniform float _Float14;
		uniform float _Float15;
		uniform float _Float8;
		uniform float _Float9;
		uniform float _Float7;


		float3 mod3D289( float3 x ) { return x - floor( x / 289.0 ) * 289.0; }

		float4 mod3D289( float4 x ) { return x - floor( x / 289.0 ) * 289.0; }

		float4 permute( float4 x ) { return mod3D289( ( x * 34.0 + 1.0 ) * x ); }

		float4 taylorInvSqrt( float4 r ) { return 1.79284291400159 - r * 0.85373472095314; }

		float snoise( float3 v )
		{
			const float2 C = float2( 1.0 / 6.0, 1.0 / 3.0 );
			float3 i = floor( v + dot( v, C.yyy ) );
			float3 x0 = v - i + dot( i, C.xxx );
			float3 g = step( x0.yzx, x0.xyz );
			float3 l = 1.0 - g;
			float3 i1 = min( g.xyz, l.zxy );
			float3 i2 = max( g.xyz, l.zxy );
			float3 x1 = x0 - i1 + C.xxx;
			float3 x2 = x0 - i2 + C.yyy;
			float3 x3 = x0 - 0.5;
			i = mod3D289( i);
			float4 p = permute( permute( permute( i.z + float4( 0.0, i1.z, i2.z, 1.0 ) ) + i.y + float4( 0.0, i1.y, i2.y, 1.0 ) ) + i.x + float4( 0.0, i1.x, i2.x, 1.0 ) );
			float4 j = p - 49.0 * floor( p / 49.0 );  // mod(p,7*7)
			float4 x_ = floor( j / 7.0 );
			float4 y_ = floor( j - 7.0 * x_ );  // mod(j,N)
			float4 x = ( x_ * 2.0 + 0.5 ) / 7.0 - 1.0;
			float4 y = ( y_ * 2.0 + 0.5 ) / 7.0 - 1.0;
			float4 h = 1.0 - abs( x ) - abs( y );
			float4 b0 = float4( x.xy, y.xy );
			float4 b1 = float4( x.zw, y.zw );
			float4 s0 = floor( b0 ) * 2.0 + 1.0;
			float4 s1 = floor( b1 ) * 2.0 + 1.0;
			float4 sh = -step( h, 0.0 );
			float4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
			float4 a1 = b1.xzyw + s1.xzyw * sh.zzww;
			float3 g0 = float3( a0.xy, h.x );
			float3 g1 = float3( a0.zw, h.y );
			float3 g2 = float3( a1.xy, h.z );
			float3 g3 = float3( a1.zw, h.w );
			float4 norm = taylorInvSqrt( float4( dot( g0, g0 ), dot( g1, g1 ), dot( g2, g2 ), dot( g3, g3 ) ) );
			g0 *= norm.x;
			g1 *= norm.y;
			g2 *= norm.z;
			g3 *= norm.w;
			float4 m = max( 0.6 - float4( dot( x0, x0 ), dot( x1, x1 ), dot( x2, x2 ), dot( x3, x3 ) ), 0.0 );
			m = m* m;
			m = m* m;
			float4 px = float4( dot( x0, g0 ), dot( x1, g1 ), dot( x2, g2 ), dot( x3, g3 ) );
			return 42.0 * dot( m, px);
		}


		float3 HSVToRGB( float3 c )
		{
			float4 K = float4( 1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0 );
			float3 p = abs( frac( c.xxx + K.xyz ) * 6.0 - K.www );
			return c.z * lerp( K.xxx, saturate( p - K.xxx ), c.y );
		}


		void vertexDataFunc( inout appdata_full v, out Input o )
		{
			UNITY_INITIALIZE_OUTPUT( Input, o );
			float3 ase_vertex3Pos = v.vertex.xyz;
			float mulTime6 = _Time.y * ( _NoiseSpeed * 0.01 );
			float simplePerlin3D2 = snoise( floor( ( ( ase_vertex3Pos + mulTime6 ) * _NoiseScale ) )*_Float0 );
			simplePerlin3D2 = simplePerlin3D2*0.5 + 0.5;
			float smoothstepResult18 = smoothstep( _NoiseSmoothStep , 1.0 , simplePerlin3D2);
			float3 ase_vertexNormal = v.normal.xyz;
			v.vertex.xyz += ( ( smoothstepResult18 * ase_vertexNormal * _Float5 * 0.05 ) + float3( 0,0,0 ) );
			v.vertex.w = 1;
		}

		void surf( Input i , inout SurfaceOutputStandard o )
		{
			float3 ase_worldNormal = i.worldNormal;
			float3 hsvTorgb3_g5 = HSVToRGB( float3(( ( tex2D( _TextureSample0, ( ( ( mul( UNITY_MATRIX_V, float4( ase_worldNormal , 0.0 ) ).xyz * 0.5 ) + 0.5 ) * _Float10 ).xy ) * _Float17 ) + _Float3 ).r,1.0,1.0) );
			float3 gammaToLinear34 = GammaToLinearSpace( hsvTorgb3_g5 );
			float3 temp_output_12_0_g9 = ( 1.0 * gammaToLinear34 );
			float dotResult28_g9 = dot( float3(0.2126729,0.7151522,0.072175) , temp_output_12_0_g9 );
			float3 temp_cast_4 = (dotResult28_g9).xxx;
			float3 lerpResult31_g9 = lerp( temp_cast_4 , temp_output_12_0_g9 , _Float16);
			o.Albedo = lerpResult31_g9;
			float3 ase_vertex3Pos = mul( unity_WorldToObject, float4( i.worldPos , 1 ) );
			float mulTime6 = _Time.y * ( _NoiseSpeed * 0.01 );
			float simplePerlin3D2 = snoise( floor( ( ( ase_vertex3Pos + mulTime6 ) * _NoiseScale ) )*_Float0 );
			simplePerlin3D2 = simplePerlin3D2*0.5 + 0.5;
			float3 hsvTorgb3_g4 = HSVToRGB( float3(( ( simplePerlin3D2 * _Float2 ) + _Float3 ),1.0,1.0) );
			float3 gammaToLinear23 = GammaToLinearSpace( hsvTorgb3_g4 );
			float3 ase_worldPos = i.worldPos;
			float3 ase_worldViewDir = normalize( UnityWorldSpaceViewDir( ase_worldPos ) );
			float fresnelNdotV37 = dot( ase_worldNormal, ase_worldViewDir );
			float fresnelNode37 = ( 0.0 + _Float13 * pow( 1.0 - fresnelNdotV37, _Float14 ) );
			float smoothstepResult18 = smoothstep( _NoiseSmoothStep , 1.0 , simplePerlin3D2);
			float3 temp_output_12_0_g8 = ( gammaToLinear23 * smoothstepResult18 );
			float dotResult28_g8 = dot( float3(0.2126729,0.7151522,0.072175) , temp_output_12_0_g8 );
			float3 temp_cast_5 = (dotResult28_g8).xxx;
			float3 lerpResult31_g8 = lerp( temp_cast_5 , temp_output_12_0_g8 , _Float8);
			float3 temp_output_12_0_g10 = ( ( gammaToLinear34 + gammaToLinear23 + ( gammaToLinear23 * ( saturate( fresnelNode37 ) * _Float15 ) ) + lerpResult31_g8 ) * _Float9 );
			float dotResult28_g10 = dot( float3(0.2126729,0.7151522,0.072175) , temp_output_12_0_g10 );
			float3 temp_cast_6 = (dotResult28_g10).xxx;
			float3 lerpResult31_g10 = lerp( temp_cast_6 , temp_output_12_0_g10 , _Float16);
			o.Emission = lerpResult31_g10;
			float temp_output_16_0 = 1.0;
			o.Metallic = temp_output_16_0;
			o.Smoothness = _Float7;
			o.Alpha = 1;
		}

		ENDCG
		CGPROGRAM
		#pragma surface surf Standard keepalpha fullforwardshadows vertex:vertexDataFunc 

		ENDCG
		Pass
		{
			Name "ShadowCaster"
			Tags{ "LightMode" = "ShadowCaster" }
			ZWrite On
			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag
			#pragma target 5.0
			#pragma multi_compile_shadowcaster
			#pragma multi_compile UNITY_PASS_SHADOWCASTER
			#pragma skip_variants FOG_LINEAR FOG_EXP FOG_EXP2
			#include "HLSLSupport.cginc"
			#if ( SHADER_API_D3D11 || SHADER_API_GLCORE || SHADER_API_GLES || SHADER_API_GLES3 || SHADER_API_METAL || SHADER_API_VULKAN )
				#define CAN_SKIP_VPOS
			#endif
			#include "UnityCG.cginc"
			#include "Lighting.cginc"
			#include "UnityPBSLighting.cginc"
			struct v2f
			{
				V2F_SHADOW_CASTER;
				float3 worldPos : TEXCOORD1;
				float3 worldNormal : TEXCOORD2;
				UNITY_VERTEX_INPUT_INSTANCE_ID
				UNITY_VERTEX_OUTPUT_STEREO
			};
			v2f vert( appdata_full v )
			{
				v2f o;
				UNITY_SETUP_INSTANCE_ID( v );
				UNITY_INITIALIZE_OUTPUT( v2f, o );
				UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO( o );
				UNITY_TRANSFER_INSTANCE_ID( v, o );
				Input customInputData;
				vertexDataFunc( v, customInputData );
				float3 worldPos = mul( unity_ObjectToWorld, v.vertex ).xyz;
				half3 worldNormal = UnityObjectToWorldNormal( v.normal );
				o.worldNormal = worldNormal;
				o.worldPos = worldPos;
				TRANSFER_SHADOW_CASTER_NORMALOFFSET( o )
				return o;
			}
			half4 frag( v2f IN
			#if !defined( CAN_SKIP_VPOS )
			, UNITY_VPOS_TYPE vpos : VPOS
			#endif
			) : SV_Target
			{
				UNITY_SETUP_INSTANCE_ID( IN );
				Input surfIN;
				UNITY_INITIALIZE_OUTPUT( Input, surfIN );
				float3 worldPos = IN.worldPos;
				half3 worldViewDir = normalize( UnityWorldSpaceViewDir( worldPos ) );
				surfIN.worldPos = worldPos;
				surfIN.worldNormal = IN.worldNormal;
				SurfaceOutputStandard o;
				UNITY_INITIALIZE_OUTPUT( SurfaceOutputStandard, o )
				surf( surfIN, o );
				#if defined( CAN_SKIP_VPOS )
				float2 vpos = IN.pos;
				#endif
				SHADOW_CASTER_FRAGMENT( IN )
			}
			ENDCG
		}
	}
	Fallback "Diffuse"
	CustomEditor "ASEMaterialInspector"
}
/*ASEBEGIN
Version=19303
Node;AmplifyShaderEditor.RangedFloatNode;12;-2864,608;Inherit;False;Property;_NoiseSpeed;Noise Speed;4;0;Create;True;0;0;0;False;0;False;0;0.52;0;0;0;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;20;-2624,512;Inherit;False;2;2;0;FLOAT;0;False;1;FLOAT;0.01;False;1;FLOAT;0
Node;AmplifyShaderEditor.PosVertexDataNode;3;-2672,176;Inherit;False;0;0;5;FLOAT3;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.SimpleTimeNode;6;-2656,352;Inherit;False;1;0;FLOAT;1;False;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleAddOpNode;5;-2400,208;Inherit;False;2;2;0;FLOAT3;0,0,0;False;1;FLOAT;0;False;1;FLOAT3;0
Node;AmplifyShaderEditor.RangedFloatNode;42;-2384,368;Inherit;False;Property;_NoiseScale;Noise Scale;11;0;Create;True;0;0;0;False;0;False;0;150;0;0;0;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;41;-2144,272;Inherit;False;2;2;0;FLOAT3;0,0,0;False;1;FLOAT;0;False;1;FLOAT3;0
Node;AmplifyShaderEditor.RangedFloatNode;4;-2064,672;Inherit;False;Property;_Float0;Float 0;0;0;Create;True;0;0;0;False;0;False;0;15.85;0;0;0;1;FLOAT;0
Node;AmplifyShaderEditor.FloorOpNode;40;-2016,272;Inherit;False;1;0;FLOAT3;0,0,0;False;1;FLOAT3;0
Node;AmplifyShaderEditor.FunctionNode;24;-1488,-832;Inherit;False;MatcapUV;-1;;3;fb12f65be31a5374bb7f51daae7d4e67;0;0;1;FLOAT3;0
Node;AmplifyShaderEditor.NoiseGeneratorNode;2;-1792,304;Inherit;True;Simplex3D;True;False;2;0;FLOAT3;0,0,0;False;1;FLOAT;1;False;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;9;-928,-384;Inherit;False;Property;_Float2;Float 2;2;0;Create;True;0;0;0;False;0;False;0;0.34;0;0;0;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;28;-1184,-688;Inherit;False;Property;_Float10;Float 10;10;0;Create;True;0;0;0;False;0;False;0;1;0;0;0;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;27;-1088,-816;Inherit;False;2;2;0;FLOAT3;0,0,0;False;1;FLOAT;0;False;1;FLOAT3;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;8;-688,-464;Inherit;False;2;2;0;FLOAT;0;False;1;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;11;-784,-272;Inherit;False;Property;_Float3;Float 3;3;0;Create;True;0;0;0;False;0;False;0;0.5214803;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;46;-640,-96;Inherit;False;Property;_Float13;Float 13;12;0;Create;True;0;0;0;False;0;False;0;1.99;0;0;0;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;47;-608,0;Inherit;False;Property;_Float14;Float 13;13;0;Create;True;0;0;0;False;0;False;0;10.65;0;0;0;1;FLOAT;0
Node;AmplifyShaderEditor.SamplerNode;26;-496,-864;Inherit;True;Property;_TextureSample0;Texture Sample 0;9;0;Create;True;0;0;0;False;0;False;-1;None;65a8f91c782020140874f3cd68a2941e;True;0;False;white;Auto;False;Object;-1;Auto;Texture2D;8;0;SAMPLER2D;;False;1;FLOAT2;0,0;False;2;FLOAT;0;False;3;FLOAT2;0,0;False;4;FLOAT2;0,0;False;5;FLOAT;1;False;6;FLOAT;0;False;7;SAMPLERSTATE;;False;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.RangedFloatNode;109;-396.4712,-619.7545;Inherit;False;Property;_Float17;Float 2;1;0;Create;True;0;0;0;False;0;False;0;0.22;0;0;0;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleAddOpNode;10;-544,-432;Inherit;False;2;2;0;FLOAT;0;False;1;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.FresnelNode;37;-400,-96;Inherit;False;Standard;WorldNormal;ViewDir;False;False;5;0;FLOAT3;0,0,1;False;4;FLOAT3;0,0,0;False;1;FLOAT;0;False;2;FLOAT;1;False;3;FLOAT;5;False;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;30;-112,-768;Inherit;False;2;2;0;COLOR;0,0,0,0;False;1;FLOAT;0;False;1;COLOR;0
Node;AmplifyShaderEditor.FunctionNode;7;-384,-400;Inherit;True;Simple HUE;-1;;4;32abb5f0db087604486c2db83a2e817a;0;1;1;FLOAT;0;False;4;FLOAT3;6;FLOAT;7;FLOAT;5;FLOAT;8
Node;AmplifyShaderEditor.RangedFloatNode;19;-432,512;Inherit;False;Property;_NoiseSmoothStep;Noise SmoothStep;7;0;Create;True;0;0;0;False;0;False;0.54;0.5845131;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.SaturateNode;49;-160,-80;Inherit;False;1;0;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;50;-192,48;Inherit;False;Property;_Float15;Float 15;14;0;Create;True;0;0;0;False;0;False;0;4.24;0;0;0;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleAddOpNode;32;-128,-640;Inherit;False;2;2;0;COLOR;0,0,0,0;False;1;FLOAT;0;False;1;COLOR;0
Node;AmplifyShaderEditor.GammaToLinearNode;23;-32,-336;Inherit;False;0;1;0;FLOAT3;0,0,0;False;1;FLOAT3;0
Node;AmplifyShaderEditor.SmoothstepOpNode;18;-32,320;Inherit;True;3;0;FLOAT;0;False;1;FLOAT;0;False;2;FLOAT;1;False;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;48;-32,-96;Inherit;False;2;2;0;FLOAT;0;False;1;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.FunctionNode;29;192,-736;Inherit;True;Simple HUE;-1;;5;32abb5f0db087604486c2db83a2e817a;0;1;1;COLOR;0,0,0,0;False;4;FLOAT3;6;FLOAT;7;FLOAT;5;FLOAT;8
Node;AmplifyShaderEditor.RangedFloatNode;117;112,96;Inherit;False;Property;_Float8;Float 8;17;0;Create;True;0;0;0;False;0;False;2;4;0;0;0;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;114;336,96;Inherit;False;2;2;0;FLOAT3;0,0,0;False;1;FLOAT;0;False;1;FLOAT3;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;38;144,-128;Inherit;True;2;2;0;FLOAT3;0,0,0;False;1;FLOAT;0;False;1;FLOAT3;0
Node;AmplifyShaderEditor.GammaToLinearNode;34;448,-704;Inherit;True;0;1;0;FLOAT3;0,0,0;False;1;FLOAT3;0
Node;AmplifyShaderEditor.FunctionNode;116;384,-32;Inherit;False;Saturation;-1;;8;4f383aa3b2a7ef640be83276d286e709;0;2;12;FLOAT3;0,0,0;False;21;FLOAT;0.5;False;1;FLOAT3;0
Node;AmplifyShaderEditor.NormalVertexDataNode;13;240,416;Inherit;False;0;5;FLOAT3;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.RangedFloatNode;15;304,560;Inherit;False;Property;_Float5;Float 5;5;0;Create;True;0;0;0;False;0;False;0;0.04;0;0;0;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;36;304,640;Inherit;False;Constant;_Float11;Float 11;10;0;Create;True;0;0;0;False;0;False;0.05;0;0;0;0;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;22;592,-256;Inherit;False;Property;_Float9;Float 9;8;0;Create;True;0;0;0;False;0;False;0;0.97;0;0;0;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleAddOpNode;35;368,-352;Inherit;False;4;4;0;FLOAT3;0,0,0;False;1;FLOAT3;0,0,0;False;2;FLOAT3;0,0,0;False;3;FLOAT3;0,0,0;False;1;FLOAT3;0
Node;AmplifyShaderEditor.RangedFloatNode;16;784,-64;Inherit;False;Constant;_Float6;Float 6;5;0;Create;True;0;0;0;False;0;False;1;0;0;0;0;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;14;688,416;Inherit;False;4;4;0;FLOAT;0;False;1;FLOAT3;0,0,0;False;2;FLOAT;0;False;3;FLOAT;0;False;1;FLOAT3;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;21;576,-160;Inherit;False;2;2;0;FLOAT3;0,0,0;False;1;FLOAT;0;False;1;FLOAT3;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;33;928,-432;Inherit;False;2;2;0;FLOAT;0;False;1;FLOAT3;0,0,0;False;1;FLOAT3;0
Node;AmplifyShaderEditor.RangedFloatNode;113;864,-272;Inherit;False;Property;_Float16;Float 16;16;0;Create;True;0;0;0;False;0;False;0;0.8260869;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.VertexColorNode;119;688,128;Inherit;False;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;120;960,144;Inherit;False;2;2;0;FLOAT;0;False;1;FLOAT;1.1;False;1;FLOAT;0
Node;AmplifyShaderEditor.ToggleSwitchNode;105;-1184,80;Inherit;False;Property;_ToggleSwitch0;Toggle Switch0;15;0;Create;True;0;0;0;False;0;False;1;True;2;0;FLOAT;0;False;1;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.PosVertexDataNode;102;224,240;Inherit;False;0;0;5;FLOAT3;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.TexCoordVertexDataNode;106;80,-1024;Inherit;False;2;4;0;5;FLOAT4;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.RangedFloatNode;17;672,32;Inherit;False;Property;_Float7;Float 7;6;0;Create;True;0;0;0;False;0;False;0;1;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleAddOpNode;101;944,384;Inherit;False;2;2;0;FLOAT3;0,0,0;False;1;FLOAT3;0,0,0;False;1;FLOAT3;0
Node;AmplifyShaderEditor.FloorOpNode;107;-800,-672;Inherit;False;1;0;FLOAT3;0,0,0;False;1;FLOAT3;0
Node;AmplifyShaderEditor.FunctionNode;111;1152,-352;Inherit;False;Saturation;-1;;9;4f383aa3b2a7ef640be83276d286e709;0;2;12;FLOAT3;0,0,0;False;21;FLOAT;0.5;False;1;FLOAT3;0
Node;AmplifyShaderEditor.FunctionNode;112;1152,-128;Inherit;False;Saturation;-1;;10;4f383aa3b2a7ef640be83276d286e709;0;2;12;FLOAT3;0,0,0;False;21;FLOAT;0.5;False;1;FLOAT3;0
Node;AmplifyShaderEditor.OneMinusNode;110;496,128;Inherit;True;1;0;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.DitheringNode;118;1264,144;Inherit;False;1;False;4;0;FLOAT;0;False;1;SAMPLER2D;;False;2;FLOAT4;0,0,0,0;False;3;SAMPLERSTATE;;False;1;FLOAT;0
Node;AmplifyShaderEditor.StandardSurfaceOutputNode;0;1584,-144;Float;False;True;-1;7;ASEMaterialInspector;0;0;Standard;FATALERROR/iridescent neon;False;False;False;False;False;False;False;False;False;False;False;False;False;False;True;False;False;False;False;False;False;Back;0;False;;0;False;;False;0;False;;0;False;;False;0;Opaque;0.5;True;True;0;False;Opaque;;Geometry;All;12;all;True;True;True;True;0;False;;False;0;False;;255;False;;255;False;;0;False;;0;False;;0;False;;0;False;;0;False;;0;False;;0;False;;0;False;;False;2;15;10;25;False;0.5;True;0;5;False;;10;False;;0;0;False;;0;False;;0;False;;0;False;;0;False;0;0,0,0,0;VertexOffset;True;False;Cylindrical;False;True;Relative;0;;-1;-1;-1;-1;0;False;0;0;False;;-1;0;False;;4;Pragma;instancing_options procedural:vertInstancingSetup;False;;Custom;False;0;0;;Define;UNITY_PARTICLE_INSTANCE_DATA MyParticleInstanceData;False;;Custom;False;0;0;;Custom;struct MyParticleInstanceData{float3x4 transform@uint color@float animFrame@float3 noise@}@;False;;Custom;False;0;0;;Include;UnityStandardParticleInstancing.cginc;False;;Custom;False;0;0;;0;0;False;0.1;False;;0;False;;False;17;0;FLOAT3;0,0,0;False;1;FLOAT3;0,0,0;False;2;FLOAT3;0,0,0;False;3;FLOAT;0;False;4;FLOAT;0;False;5;FLOAT;0;False;6;FLOAT3;0,0,0;False;7;FLOAT3;0,0,0;False;8;FLOAT;0;False;9;FLOAT;0;False;10;FLOAT;0;False;13;FLOAT3;0,0,0;False;11;FLOAT3;0,0,0;False;12;FLOAT3;0,0,0;False;16;FLOAT4;0,0,0,0;False;14;FLOAT4;0,0,0,0;False;15;FLOAT3;0,0,0;False;0
WireConnection;20;0;12;0
WireConnection;6;0;20;0
WireConnection;5;0;3;0
WireConnection;5;1;6;0
WireConnection;41;0;5;0
WireConnection;41;1;42;0
WireConnection;40;0;41;0
WireConnection;2;0;40;0
WireConnection;2;1;4;0
WireConnection;27;0;24;0
WireConnection;27;1;28;0
WireConnection;8;0;2;0
WireConnection;8;1;9;0
WireConnection;26;1;27;0
WireConnection;10;0;8;0
WireConnection;10;1;11;0
WireConnection;37;2;46;0
WireConnection;37;3;47;0
WireConnection;30;0;26;0
WireConnection;30;1;109;0
WireConnection;7;1;10;0
WireConnection;49;0;37;0
WireConnection;32;0;30;0
WireConnection;32;1;11;0
WireConnection;23;0;7;6
WireConnection;18;0;2;0
WireConnection;18;1;19;0
WireConnection;48;0;49;0
WireConnection;48;1;50;0
WireConnection;29;1;32;0
WireConnection;114;0;23;0
WireConnection;114;1;18;0
WireConnection;38;0;23;0
WireConnection;38;1;48;0
WireConnection;34;0;29;6
WireConnection;116;12;114;0
WireConnection;116;21;117;0
WireConnection;35;0;34;0
WireConnection;35;1;23;0
WireConnection;35;2;38;0
WireConnection;35;3;116;0
WireConnection;14;0;18;0
WireConnection;14;1;13;0
WireConnection;14;2;15;0
WireConnection;14;3;36;0
WireConnection;21;0;35;0
WireConnection;21;1;22;0
WireConnection;33;0;16;0
WireConnection;33;1;34;0
WireConnection;120;0;119;4
WireConnection;101;0;14;0
WireConnection;111;12;33;0
WireConnection;111;21;113;0
WireConnection;112;12;21;0
WireConnection;112;21;113;0
WireConnection;118;0;120;0
WireConnection;0;0;111;0
WireConnection;0;2;112;0
WireConnection;0;3;16;0
WireConnection;0;4;17;0
WireConnection;0;11;101;0
ASEEND*/
//CHKSM=4BFEB7D06B5FCF32692CAE4FFB1DD8D9CDBCC784